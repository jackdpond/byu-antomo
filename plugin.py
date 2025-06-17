from magicgui import magicgui, widgets
from napari.types import ImageData, LabelsData, PointsData
from napari.layers import Labels, Points
from napari.utils.notifications import show_info, show_error
import numpy as np
import pandas as pd
import datetime
import os
import mrcfile
from skimage import exposure
import json
import time
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt
from skimage.transform import downscale_local_mean
import dask.array as da

CONFIG_PATH = os.path.expanduser('~/.napari_plugin_config.json')

def get_tomogram_source_path(viewer):
    """Get the source path of the tomogram layer."""
    for layer in viewer.layers:
        if layer.name == "Tomogram":
            # Try to get path from metadata first
            if hasattr(layer, 'metadata') and layer.metadata:
                source_path = layer.metadata.get('source')
                if source_path:
                    return os.path.abspath(source_path) if not os.path.isabs(source_path) else source_path
            
            # Fallback to source.path
            if hasattr(layer, 'source') and hasattr(layer.source, 'path'):
                source_path = layer.source.path
                if source_path:
                    return source_path
    return "unknown"

def load_and_stretch(filepath):
    """
    Load tomogram as a Dask array and apply per-slice contrast stretching for fast, clear display.
    Returns a Dask array suitable for napari.
    """
    with mrcfile.mmap(filepath, permissive=True) as mrc:
        data = mrc.data  # This is a numpy memmap array
        # Wrap as Dask array, chunked by z-slice
        dask_data = da.from_array(data, chunks=(1, data.shape[1], data.shape[2]))

    def stretch_slice(slice2d):
        # Compute 2nd and 98th percentiles for this slice
        p2, p98 = np.percentile(slice2d, (2, 98))
        # Avoid degenerate case
        if p2 == p98:
            return np.zeros_like(slice2d, dtype=np.float32)
        # Stretch
        return exposure.rescale_intensity(slice2d, in_range=(p2, p98)).astype(np.float32)

    # Map the stretching function over each z-slice
    stretched = dask_data.map_blocks(
        lambda block: np.stack([stretch_slice(s) for s in block]),
        dtype=np.float32,
        chunks=dask_data.chunks
    )
    return stretched

def get_reduction_factors():
    return (
        int(plugin_config.get('tomogram_z_step', 2)),
        int(plugin_config.get('tomogram_y_step', 1)),
        int(plugin_config.get('tomogram_x_step', 1)),
    )

def get_tomogram_layer(viewer):
    for lyr in viewer.layers:
        if lyr.name == 'Tomogram' and hasattr(lyr, 'data') and hasattr(lyr.data, 'shape'):
            return lyr
    return None

def create_point_layer(viewer, label, color):
    """Create a points layer for point-based annotations."""
    name = f"{label}_points"
    if name not in viewer.layers:
        layer = viewer.add_points(name=name, ndim=3, size=8, face_color=color)
        layer.mode = 'add'
        viewer.layers.selection.active = layer
        return layer
    return viewer.layers[name]

def create_mask_layer(viewer, label, shape):
    """Create a mask or shapes layer for mask-based annotations."""
    name = f"{label}_mask"
    if name not in viewer.layers:
        if label == "chemosensory_array":
            mask = np.zeros(shape, dtype=np.uint8)
            layer = viewer.add_labels(mask, name=name)
            layer.mode = 'paint'
            layer.selected_label = 3
            layer.brush_size = 20
        else:
            layer = viewer.add_shapes(
                name=name, 
                ndim=3, 
                shape_type='rectangle',
                edge_color='magenta',
                face_color='magenta',
                opacity=0.4
            )
            layer.mode = 'add_rectangle'
        viewer.layers.selection.active = layer
        return layer
    return viewer.layers[name]

def create_annotation_widget(viewer, config_refresh_callback=None):
    container = widgets.Container(layout='vertical')
    
    # Create shared inputs
    label_widget = widgets.ComboBox(
        name='label',
        choices=["pilus", "flagellar_motor", "chemosensory_array", "ribosome", "cell", "storage_granule"],
        label='Label'
    )
    user_widget = widgets.LineEdit(
        name='user',
        label='User'
    )
    
    container.extend([label_widget, user_widget])
    
    def add_annotation():
        label = label_widget.value
        user = user_widget.value
        
        if label in ["pilus", "flagellar_motor", "ribosome"]:
            color = {
                'pilus': 'red',
                'flagellar_motor': 'blue',
                'ribosome': 'green'
            }[label]
            layer = create_point_layer(viewer, label, color)
        else:
            image_layer = get_tomogram_layer(viewer)
            if image_layer is None:
                show_error("⚠️ Could not find Tomogram layer to determine mask shape.")
                return
            layer = create_mask_layer(viewer, label, image_layer.data.shape)
        
        show_info(f"✅ {label} annotation layer added successfully")
    
    add_button = widgets.PushButton(text="Add Annotation", name='add_button')
    add_button.clicked.connect(add_annotation)
    
    def save_annotation():
        label = label_widget.value
        user = user_widget.value
        
        # Check if the annotation layer exists
        layer_name = f"{label}_points" if label in ["pilus", "flagellar_motor", "ribosome"] else f"{label}_mask"
        if layer_name not in viewer.layers:
            show_error(f"⚠️ No {label} annotation layer found. Please add an annotation layer before saving.")
            return

        timestamp = datetime.datetime.now().isoformat()
        source_path = get_tomogram_source_path(viewer)
        
        # Use config for annotation directory
        shared_dir = plugin_config['annotation_dir']
        os.makedirs(shared_dir, exist_ok=True)

        # Define column order for each label
        columns_dict = {
            "pilus":      ["z", "y", "x", "label", "user", "timestamp", "source_path"],
            "flagellar_motor": ["z", "y", "x", "label", "user", "timestamp", "source_path"],
            "ribosome":   ["z", "y", "x", "label", "user", "timestamp", "source_path"],
            "cell":       ["z", "y", "x", "width", "height", "label", "user", "timestamp", "source_path"],
            "storage_granule": ["z", "y", "x", "width", "height", "label", "user", "timestamp", "source_path"],
            "chemosensory_array": ["z", "label", "user", "timestamp", "source_path", "mask_path"],
        }

        z_step, y_step, x_step = get_reduction_factors()
        layer = viewer.layers[layer_name]

        if label in ["pilus", "flagellar_motor", "ribosome"]:
            coords = layer.data
            scaled_coords = coords.copy()
            scaled_coords[:, 0] *= z_step
            scaled_coords[:, 1] *= y_step
            scaled_coords[:, 2] *= x_step
            
            df = pd.DataFrame({
                "z": scaled_coords[:, 0],
                "y": scaled_coords[:, 1],
                "x": scaled_coords[:, 2],
                "label": label,
                "user": user,
                "timestamp": timestamp,
                "source_path": source_path,
            })
            df = df[columns_dict[label]]

        elif label == "chemosensory_array":
            current_z_reduced = viewer.dims.current_step[0]
            current_z_original = current_z_reduced * z_step
            mask_2d = layer.data[current_z_reduced]
            
            original_shape = (mask_2d.shape[0] * y_step, mask_2d.shape[1] * x_step)
            scaled_mask = np.zeros(original_shape, dtype=np.uint8)
            
            # Scale up the mask coordinates
            for i in range(mask_2d.shape[0]):
                for j in range(mask_2d.shape[1]):
                    if mask_2d[i, j] != 0:
                        scaled_mask[i * y_step, j * x_step] = 1
            
            # Fill in the gaps created by scaling
            if y_step > 1:
                scaled_mask = fill_mask_gaps(scaled_mask, y_step)
            if x_step > 1:
                scaled_mask = fill_mask_gaps(scaled_mask.T, x_step).T
            
            mask_path = os.path.join(shared_dir, f"{label}_{user}_{timestamp}_slice_{current_z_original:03d}.npy")
            np.save(mask_path, scaled_mask)
            
            df = pd.DataFrame([{
                "z": current_z_original,
                "label": label,
                "user": user,
                "timestamp": timestamp,
                "source_path": source_path,
                "mask_path": mask_path
            }])
            df = df[columns_dict[label]]
            
        elif label in ["cell", "storage_granule"]:
            current_z_reduced = viewer.dims.current_step[0]
            current_z_original = current_z_reduced * z_step
            shapes_layer = layer
            rows = []
            
            for poly in shapes_layer.data:
                if np.allclose(poly[:, 0], current_z_reduced):
                    poly_yx = poly[:, 1:3]
                    min_y, max_y = np.min(poly_yx[:, 0]), np.max(poly_yx[:, 0])
                    min_x, max_x = np.min(poly_yx[:, 1]), np.max(poly_yx[:, 1])
                    
                    center_y = ((min_y + max_y) / 2) * y_step
                    center_x = ((min_x + max_x) / 2) * x_step
                    width = (max_x - min_x) * x_step
                    height = (max_y - min_y) * y_step
                    
                    rows.append({
                        "z": current_z_original,
                        "y": center_y,
                        "x": center_x,
                        "width": width,
                        "height": height,
                        "label": label,
                        "user": user,
                        "timestamp": timestamp,
                        "source_path": source_path,
                    })
            
            df = pd.DataFrame(rows)
            df = df[columns_dict[label]]
        
        # Save to CSV
        csv_path = os.path.join(shared_dir, f"{label}_annotations.csv")
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        
        show_info(f"✅ Saved {len(df)} {label} annotations")
    
    save_button = widgets.PushButton(text="Save Annotation", name='save_button')
    save_button.clicked.connect(save_annotation)
    
    container.extend([add_button, save_button])
    return container

def create_annotation_viewer_widget(viewer, config_refresh_callback=None):
    # Create the main container with vertical layout
    container = widgets.Container(layout='vertical')
    
    # Add title
    title = widgets.Label(value="Saved Annotations")
    container.append(title)
    
    # Add refresh button
    refresh_button = widgets.PushButton(text="Refresh Stats")
    container.append(refresh_button)
    
    # Create a sub-container for stats
    stats_container = widgets.Container(layout='vertical')
    container.append(stats_container)
    
    def update_stats():
        # Clear only the stats container
        while len(stats_container) > 0:
            stats_container.pop()
        
        # Get current tomogram source
        source_path = None
        for layer in viewer.layers:
            if layer.name == "Tomogram":
                if hasattr(layer, 'metadata') and layer.metadata:
                    source_path = layer.metadata.get('source')
                    if source_path and not os.path.isabs(source_path):
                        source_path = os.path.abspath(source_path)
                break
        
        if not source_path or source_path == "unknown":
            stats_container.append(widgets.Label(value="No tomogram source found"))
            return
        
        # Use config for annotation directory
        shared_dir = plugin_config['annotation_dir']
        import glob
        csv_files = glob.glob(os.path.join(shared_dir, "*_annotations.csv"))
        if not csv_files:
            stats_container.append(widgets.Label(value="No annotations found"))
            return
        
        # Aggregate all annotations for this tomogram
        all_rows = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if 'source_path' in df.columns:
                    df_source = df[df['source_path'] == source_path]
                    if not df_source.empty:
                        all_rows.append(df_source)
            except Exception as e:
                stats_container.append(widgets.Label(value=f"Error reading {os.path.basename(csv_path)}: {str(e)}"))
        if not all_rows:
            stats_container.append(widgets.Label(value="No annotations for this tomogram"))
            return
        df_source = pd.concat(all_rows, ignore_index=True)
        # Show all annotations as clickable buttons
        for idx, row in df_source.iterrows():
            label = row['label']
            label_container = widgets.Container(layout='horizontal')
            label_container.style = {'margin': '2px 0'}  # Reduce vertical spacing
            label_name = widgets.Label(value=f"{label}:")
            label_name.min_width = 100
            label_container.append(label_name)
            # Create a button for this annotation
            btn = widgets.PushButton(text="Show", name=f"show_{idx}")
            def make_on_click(row=row, label=label, idx=idx):
                def on_click():
                    # Remove any previous display layer for this annotation
                    display_layer_name = f"[DISPLAY] {label} {idx}"
                    if display_layer_name in viewer.layers:
                        viewer.layers.remove(display_layer_name)
                    if label in ["pilus", "flagellar_motor", "ribosome"]:
                        z_step, y_step, x_step = get_reduction_factors()
                        coords = np.array([[row['z'] / z_step, row['y'] / y_step, row['x'] / x_step]])
                        layer = viewer.add_points(coords, name=display_layer_name, ndim=3, size=10, face_color='yellow')
                        layer.mode = 'pan_zoom'  # Not editable
                        layer.editable = False
                        z = int(row['z'] / z_step)
                        jump_to_z_slice(viewer, z)
                    elif label == "chemosensory_array" and pd.notna(row['mask_path']):
                        z_step, y_step, x_step = get_reduction_factors()
                        mask_2d = np.load(row['mask_path'])
                        from skimage.transform import downscale_local_mean
                        reduced_mask = mask_2d
                        if y_step > 1 or x_step > 1:
                            reduced_mask = downscale_local_mean(mask_2d, (y_step, x_step))
                            reduced_mask = (reduced_mask > 0.5).astype(np.uint8)  # Binarize
                        tomogram_shape = viewer.layers['Tomogram'].data.shape
                        mask_3d = np.zeros(tomogram_shape, dtype=np.uint8)
                        z_slice = int(row['z'] / z_step)
                        # Debugging statements
                        print("[DEBUG] Tomogram shape:", tomogram_shape)
                        print("[DEBUG] Reduced mask shape:", reduced_mask.shape)
                        print("[DEBUG] Assigning to z-slice:", z_slice)
                        print("[DEBUG] Mask unique values:", np.unique(reduced_mask))
                        mask_3d[z_slice] = reduced_mask
                        layer = viewer.add_labels(mask_3d, name=display_layer_name)
                        layer.mode = 'pan_zoom'  # Not editable
                        layer.editable = False
                        jump_to_z_slice(viewer, z_slice)
                    elif label in ["cell", "storage_granule"] and pd.notna(row['z']) and pd.notna(row['y']) and pd.notna(row['x']) and pd.notna(row['width']) and pd.notna(row['height']):
                        z_step, y_step, x_step = get_reduction_factors()
                        z = row['z'] / z_step
                        y = row['y'] / y_step
                        x = row['x'] / x_step
                        width = row['width'] / x_step
                        height = row['height'] / y_step
                        y0 = y - height / 2
                        y1 = y + height / 2
                        x0 = x - width / 2
                        x1 = x + width / 2
                        rect = np.array([
                            [z, y0, x0],
                            [z, y0, x1],
                            [z, y1, x1],
                            [z, y1, x0],
                        ])
                        layer = viewer.add_shapes([rect], name=display_layer_name, shape_type='polygon', edge_color='yellow', face_color='yellow', opacity=0.4)
                        layer.mode = 'pan_zoom'  # Not editable
                        layer.editable = False
                        jump_to_z_slice(viewer, int(z))
                    else:
                        show_error(f"Cannot display annotation: {label}")
                return on_click
            btn.clicked.connect(make_on_click())
            label_container.append(btn)
            stats_container.append(label_container)
    
    refresh_button.clicked.connect(update_stats)
    
    # Initial update
    update_stats()
    
    # Add config button at the bottom right
    config_row = widgets.Container(layout='horizontal')
    config_row.append(widgets.Label(value=""))  # Spacer
    config_button = widgets.PushButton(text="\u2699")
    config_button.min_width = 32
    config_button.max_width = 32
    def open_config():
        dialog = create_config_dialog(refresh_callbacks=[lambda: config_refresh_callback() if config_refresh_callback else None])
        dialog.show()
    config_button.clicked.connect(open_config)
    config_row.append(config_button)
    container.append(config_row)

    return container

def load_plugin_config():
    # Load config from disk or return defaults
    default = {
        'annotation_dir': os.path.expanduser('~/groups/fslg_imagseg/jackson/Napari/Annotations'),
        'tomo_ids_csv': os.path.expanduser('~/groups/fslg_imagseg/jackson/Napari/tomo_ids.csv'),
        'tomogram_z_step': 2,  # Default to taking every second slice in Z
        'tomogram_y_step': 1,  # Default to no skipping in Y
        'tomogram_x_step': 1   # Default to no skipping in X
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
                default.update(cfg)
        except Exception:
            pass
    return default

def save_plugin_config(cfg):
    # Convert all values to strings (handles Path objects)
    cfg_str = {k: str(v) for k, v in cfg.items()}
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg_str, f)
    except Exception as e:
        print(f"Failed to save config: {e}")

plugin_config = load_plugin_config()

def create_config_dialog(refresh_callbacks=None):
    dialog = widgets.Container(layout='vertical')
    dialog.max_width = 600  # Make the dialog wider to fit everything
    
    # Main settings
    dialog.append(widgets.Label(value="Plugin Settings"))
    annotation_dir_edit = widgets.FileEdit(mode='d', value=plugin_config['annotation_dir'], label="Annotation Directory")
    tomo_ids_edit = widgets.FileEdit(mode='r', value=plugin_config['tomo_ids_csv'], label="Tomo IDs CSV")
    
    # Store default values
    default_steps = {
        'z': 2,
        'y': 1,
        'x': 1
    }
    
    # Create step controls in a horizontal layout
    steps_row = widgets.Container(layout='horizontal')
    steps_row.style = {'margin': '5px 0'}  # Add some vertical margin
    
    # Label on the left
    steps_label = widgets.Label(value="Tomogram Steps:")
    steps_label.min_width = 150  # Match width with other labels
    steps_row.append(steps_label)
    
    # Step controls in the middle
    z_step_edit = widgets.SpinBox(
        value=plugin_config.get('tomogram_z_step', default_steps['z']),
        min=1,
        max=10,
        label="Z"
    )
    y_step_edit = widgets.SpinBox(
        value=plugin_config.get('tomogram_y_step', default_steps['y']),
        min=1,
        max=10,
        label="Y"
    )
    x_step_edit = widgets.SpinBox(
        value=plugin_config.get('tomogram_x_step', default_steps['x']),
        min=1,
        max=10,
        label="X"
    )
    
    # Add change handlers for step controls
    def on_step_change():
        if (z_step_edit.value != plugin_config.get('tomogram_z_step', default_steps['z']) or
            y_step_edit.value != plugin_config.get('tomogram_y_step', default_steps['y']) or
            x_step_edit.value != plugin_config.get('tomogram_x_step', default_steps['x'])):
            show_info("⚠️ Changing tomogram steps will affect the resolution of newly loaded tomograms. Existing annotations will not be affected.")
    
    z_step_edit.changed.connect(on_step_change)
    y_step_edit.changed.connect(on_step_change)
    x_step_edit.changed.connect(on_step_change)
    
    # Add step controls to the row
    steps_row.extend([z_step_edit, y_step_edit, x_step_edit])
    
    # Add reset button on the right
    def reset_steps():
        z_step_edit.value = default_steps['z']
        y_step_edit.value = default_steps['y']
        x_step_edit.value = default_steps['x']
        show_info("✅ Tomogram steps reset to defaults")
    
    reset_button = widgets.PushButton(text="Reset")
    reset_button.min_width = 80
    reset_button.clicked.connect(reset_steps)
    steps_row.append(reset_button)
    
    # Add all widgets to dialog
    dialog.append(annotation_dir_edit)
    dialog.append(tomo_ids_edit)
    dialog.append(steps_row)
    
    # Save button
    save_button = widgets.PushButton(text="Save")
    
    def save_settings():
        plugin_config['annotation_dir'] = annotation_dir_edit.value
        plugin_config['tomo_ids_csv'] = tomo_ids_edit.value
        plugin_config['tomogram_z_step'] = z_step_edit.value
        plugin_config['tomogram_y_step'] = y_step_edit.value
        plugin_config['tomogram_x_step'] = x_step_edit.value
        save_plugin_config(plugin_config)
        # Call refresh callbacks if provided
        if refresh_callbacks:
            for cb in refresh_callbacks:
                cb()
        show_info("Plugin settings saved and widgets refreshed.")
        dialog.hide()
    
    save_button.clicked.connect(save_settings)
    dialog.append(save_button)
    
    return dialog

def create_tomogram_navigator_widget(viewer, saved_annotations_widget=None, config_refresh_callback=None):
    import csv
    # Main container
    container = widgets.Container(layout='vertical')
    container.max_width = 300
    title = widgets.Label(value="Tomogram Navigator")
    title.max_width = 300
    container.append(title)

    # Read tomo_ids.csv from config
    tomo_csv = plugin_config['tomo_ids_csv']
    tomo_ids = []
    file_paths = []
    if os.path.exists(tomo_csv):
        with open(tomo_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tomo_ids.append(row['tomo_id'])
                file_paths.append(row['file_path'])
    else:
        container.append(widgets.Label(value="No tomo_ids.csv found"))
        return container

    # Read annotation summary for preview dots from config
    ann_csv = os.path.join(plugin_config['annotation_dir'], "annotations_all.csv")
    ann_df = None
    if os.path.exists(ann_csv):
        ann_df = pd.read_csv(ann_csv)

    # Dropdown for tomo_id selection
    tomo_dropdown = widgets.ComboBox(choices=tomo_ids, label="Tomo ID")
    tomo_dropdown.max_width = 280
    container.append(tomo_dropdown)

    # Preview area
    preview_container = widgets.Container(layout='vertical')
    preview_container.max_width = 280
    preview_container.style = {'margin': '0'}  # Remove vertical margin
    container.append(preview_container)

    def update_preview():
        while len(preview_container) > 0:
            preview_container.pop()
        selected_id = tomo_dropdown.value
        if selected_id not in tomo_ids:
            return
        idx = tomo_ids.index(selected_id)
        tomo_id = tomo_ids[idx]
        file_path = file_paths[idx]
        if ann_df is not None:
            ann_rows = ann_df[ann_df['source_path'] == file_path]
            dot_row = widgets.Container(layout='horizontal')
            dot_row.max_width = 260
            dot_row.style = {'margin': '0'}
            for label, color in zip([
                'pilus', 'flagellar_motor', 'chemosensory_array'],
                ['#e74c3c', '#3498db', '#9b59b6']):
                count = (ann_rows['label'] == label).sum()
                for _ in range(count):
                    dot = widgets.Label(value="  ")
                    dot.style = {'background': color, 'border-radius': '8px', 'min-width': '16px', 'min-height': '16px', 'margin': '2px'}
                    dot.max_width = 16
                    dot_row.append(dot)
            preview_container.append(dot_row)

    tomo_dropdown.changed.connect(update_preview)
    update_preview()

    # Load and config buttons in a row, right-aligned
    button_row = widgets.Container(layout='horizontal')
    load_button = widgets.PushButton(text="Load Tomogram")
    load_button.max_width = 200
    def load_tomogram():
        print("Load Tomogram button pressed")  # Debug print
        start_time = time.time()
        selected_id = tomo_dropdown.value
        if selected_id not in tomo_ids:
            show_error("No tomogram selected.")
            return
        idx = tomo_ids.index(selected_id)
        file_path = file_paths[idx]
        # Remove all layers before loading the new tomogram
        while len(viewer.layers) > 0:
            viewer.layers.pop()
        try:
            # Set cursor to spinning icon
            print(f"[DEBUG] Whatever comes first in {time.time() - start_time:.2f} seconds")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            start_time = time.time()
            data = load_and_stretch(file_path)
            print(f"[DEBUG] Processed tomogram in {time.time() - start_time:.2f} seconds")
            data = rescale(data)
            viewer.add_image(data, name="Tomogram", metadata={'source': file_path})
            show_info(f"Loaded tomogram: {file_path}")
            if saved_annotations_widget is not None:
                for w in saved_annotations_widget:
                    if hasattr(w, 'text') and w.text == "Refresh Stats":
                        w.clicked.emit()
                        break
        except Exception as e:
            show_error(f"Failed to load tomogram: {e}")
        finally:
            # Reset cursor back to normal
            QApplication.restoreOverrideCursor()
    load_button.clicked.connect(load_tomogram)
    button_row.append(load_button)
    config_button = widgets.PushButton(text="\u2699")
    config_button.min_width = 32
    config_button.max_width = 32
    def open_config():
        dialog = create_config_dialog(refresh_callbacks=[lambda: config_refresh_callback() if config_refresh_callback else None])
        dialog.show()
    config_button.clicked.connect(open_config)
    button_row.append(config_button)
    container.append(button_row)

    return container

def jump_to_z_slice(viewer, z):
    # Clamp z to valid range
    z = int(z)
    max_z = viewer.layers['Tomogram'].data.shape[0] - 1
    z = max(0, min(z, max_z))
    scrolled_axis = viewer.dims.order[0]
    # Use a singleShot timer to set after the layer is added
    QTimer.singleShot(100, lambda: viewer.dims.set_current_step(scrolled_axis, z))

def fill_mask_gaps(mask, step):
    """Fill gaps in a scaled mask by interpolating between non-zero values."""
    result = mask.copy()
    for i in range(1, len(mask) - 1):
        if mask[i-1] != 0 and mask[i+1] != 0:
            result[i] = 1
    return result

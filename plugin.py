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

CONFIG_PATH = os.path.expanduser('~/.napari_plugin_config.json')

def rescale(array):
    """ Rescales array values so that all values are between 0 and 1. """
    print("[DEBUG] Starting rescale...")
    start = time.time()
    maximum = np.max(array)
    minimum = np.min(array)
    range = maximum - minimum
    result = (array - minimum) / range
    print(f"[DEBUG] Finished rescale in {time.time() - start:.2f} seconds.")
    return result

def mrc_to_np(filepath):
    """ Converts .mrc file (or .rec file) to numpy array. """
    with mrcfile.open(filepath, 'r') as mrc:
        data = mrc.data.astype(np.float64)
        return data

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

def process_tomogram(data):
    """ Simple tomogram processing - uses contrast stretching to improve contrast. """
    print("[DEBUG] Starting process_tomogram...")
    start = time.time()
    # Use the configured steps, defaulting to 1 if not set
    z_step, y_step, x_step = get_reduction_factors()
    data = data[::z_step, ::y_step, ::x_step]
    p2, p98 = np.percentile(data, (2, 98))
    print(f"[DEBUG] Percentiles computed in {time.time() - start:.2f} seconds. p2={p2}, p98={p98}")
    start2 = time.time()
    data_rescale = exposure.rescale_intensity(data, in_range=(p2, p98))
    print(f"[DEBUG] rescale_intensity finished in {time.time() - start2:.2f} seconds.")
    print(f"[DEBUG] Finished process_tomogram in {time.time() - start:.2f} seconds.")
    return data_rescale

def create_annotation_widget(viewer, config_refresh_callback=None):
    # Create the container with vertical layout
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
    
    # Add inputs to container
    container.extend([label_widget, user_widget])
    
    # Create button container with vertical layout
    button_container = widgets.Container(layout='vertical')
    
    # Add Annotation button
    def add_annotation():
        label = label_widget.value
        user = user_widget.value
        
        if label in ["pilus", "flagellar_motor", "ribosome"]:
            name = f"{label}_points"
            if name not in viewer.layers:
                point_color = (
                    'red' if label == 'pilus' else
                    'blue' if label == 'flagellar_motor' else
                    'green'  # ribosome
                )
                labels_layer = viewer.add_points(name=name, ndim=3, size=8, face_color=point_color)
                labels_layer.mode = 'add'
                # Set the layer as selected for immediate editing
                viewer.layers.selection.active = labels_layer
        elif label in ["chemosensory_array", "cell", "storage_granule"]:
            name = f"{label}_mask"
            if name not in viewer.layers:
                # Always get the Tomogram layer for shape
                image_layer = get_tomogram_layer(viewer)
                if image_layer is None:
                    show_error("⚠️ Could not find Tomogram layer to determine mask shape.")
                    return
                shape = image_layer.data.shape
                if label == "chemosensory_array":
                    mask = np.zeros(shape, dtype=np.uint8)
                    labels_layer = viewer.add_labels(mask, name=name)
                    labels_layer.mode = 'paint'  # Start in paint mode
                    labels_layer.selected_label = 3  # Set label to 1 (for painting)
                    labels_layer.brush_size = 20  # Set a reasonable brush size
                    viewer.layers.selection.active = labels_layer
                else:
                    shapes_layer = viewer.add_shapes(name=name, ndim=3, shape_type='polygon', edge_color='magenta', face_color='magenta', opacity=0.4)
                    viewer.layers.selection.active = shapes_layer
        show_info(f"✅ {label} annotation layer added successfully")
    
    add_button = widgets.PushButton(
        text="Add Annotation",
        name='add_button'
    )
    add_button.clicked.connect(add_annotation)
    
    # Save Annotation button
    def save_annotation():
        label = label_widget.value
        user = user_widget.value
        
        # Check if the annotation layer exists before saving
        layer_name = f"{label}_points" if label in ["pilus", "flagellar_motor", "ribosome"] else f"{label}_mask"
        if layer_name not in viewer.layers:
            show_error(f"⚠️ No {label} annotation layer found. Please add an annotation layer before saving.")
            return

        # Save annotation logic
        timestamp = datetime.datetime.now().isoformat()
        # Get source path from the tomogram layer
        source_path = None
        for layer in viewer.layers:
            if layer.name == "Tomogram":
                print(f"Found Tomogram layer: {layer}")
                print(f"Layer source: {layer.source}")
                print(f"Layer metadata: {layer.metadata}")
                # Try to get path from metadata first
                if hasattr(layer, 'metadata') and layer.metadata:
                    source_path = layer.metadata.get('source')
                    if source_path:
                        # Convert to absolute path if it's relative
                        if not os.path.isabs(source_path):
                            source_path = os.path.abspath(source_path)
                        print(f"Found source path in metadata: {source_path}")
                        break
                
                # Fallback to source.path if metadata doesn't have it
                if hasattr(layer, 'source') and hasattr(layer.source, 'path'):
                    source_path = layer.source.path
                    if source_path:
                        print(f"Found source path in layer.source: {source_path}")
                        break
        
        if source_path is None:
            source_path = "unknown"  # Fallback if no path is found
            print("No source path found, using 'unknown'")

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

        # Get the current step factors
        z_step, y_step, x_step = get_reduction_factors()

        if label in ["pilus", "flagellar_motor", "ribosome"]:
            layer = viewer.layers[layer_name]
            if not layer:
                show_error(f"⚠️ No layer named {layer_name}")
                return
            coords = layer.data
            # Scale coordinates by step factors to get original tomogram coordinates
            scaled_coords = coords.copy()
            scaled_coords[:, 0] *= z_step
            scaled_coords[:, 1] *= y_step
            scaled_coords[:, 2] *= x_step
            # Create DataFrame with coordinates and other columns
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
            layer = viewer.layers[layer_name]
            if not layer:
                show_error(f"⚠️ No layer named {layer_name}")
                return
            current_z_reduced = viewer.dims.current_step[0]
            current_z_original = current_z_reduced * z_step
            # Get the 2D mask from the current slice
            mask_2d = layer.data[current_z_reduced]
            # Create a new mask with the original tomogram dimensions
            original_shape = (mask_2d.shape[0] * y_step, mask_2d.shape[1] * x_step)
            scaled_mask = np.zeros(original_shape, dtype=np.uint8)
            # Scale up the mask coordinates
            for i in range(mask_2d.shape[0]):
                for j in range(mask_2d.shape[1]):
                    if mask_2d[i, j] == 1:
                        scaled_mask[i * y_step, j * x_step] = 1
            # Fill in the gaps created by scaling
            if y_step > 1:
                scaled_mask = fill_mask_gaps(scaled_mask, y_step)
            if x_step > 1:
                scaled_mask = fill_mask_gaps(scaled_mask.T, x_step).T
            # Save the scaled and filled mask with original z-slice in filename
            mask_path = os.path.join(shared_dir, f"{label}_{user}_{timestamp}_slice_{current_z_original:03d}.npy")
            np.save(mask_path, scaled_mask)
            # Create DataFrame with all columns in correct order
            df = pd.DataFrame([{
                "z": current_z_original,  # Store the original z-slice in the CSV
                "label": label,
                "user": user,
                "timestamp": timestamp,
                "source_path": source_path,
                "mask_path": mask_path
            }])
            df = df[columns_dict[label]]
        elif label in ["cell", "storage_granule"]:
            layer = viewer.layers[layer_name]
            if not layer:
                show_error(f"⚠️ No layer named {layer_name}")
                return
            current_z_reduced = viewer.dims.current_step[0]
            current_z_original = current_z_reduced * z_step
            shapes_layer = layer
            rows = []
            for poly in shapes_layer.data:
                # Only use shapes on the current z-slice
                if np.allclose(poly[:, 0], current_z_reduced):
                    # Get y, x coordinates
                    poly_yx = poly[:, 1:3]
                    min_y, max_y = np.min(poly_yx[:, 0]), np.max(poly_yx[:, 0])
                    min_x, max_x = np.min(poly_yx[:, 1]), np.max(poly_yx[:, 1])
                    # Scale center and size to original tomogram coordinates
                    center_y = ((min_y + max_y) / 2) * y_step
                    center_x = ((min_x + max_x) / 2) * x_step
                    width = (max_x - min_x) * x_step
                    height = (max_y - min_y) * y_step
                    center_z = current_z_original
                    rows.append({
                        "z": center_z,
                        "y": center_y,
                        "x": center_x,
                        "width": width,
                        "height": height,
                        "label": label,
                        "user": user,
                        "timestamp": timestamp,
                        "source_path": source_path,
                    })
            if not rows:
                show_error(f"⚠️ No shapes found on current z-slice for {label}.")
                return
            df = pd.DataFrame(rows)
            df = df[columns_dict[label]]

        else:
            show_error("⚠️ Unknown label type.")
            return

        label_path = os.path.join(shared_dir, f"{label}_annotations.csv")
        df.to_csv(label_path, mode='a', header=not os.path.exists(label_path), index=False)
        show_info(f"✅ Successfully saved {label} annotation by {user} to {label_path}")
    
    save_button = widgets.PushButton(
        text="Save Annotation",
        name='save_button'
    )
    save_button.clicked.connect(save_annotation)
    
    # Add buttons to button container
    button_container.extend([add_button, save_button])
    
    # Add button container to main container
    container.append(button_container)
    
    # Config button at the bottom right
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
            data = mrc_to_np(file_path)
            print(f"[DEBUG] Processed tomogram in {time.time() - start_time:.2f} seconds")
            data = process_tomogram(data)
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
    """Fill gaps in a binary mask created by coordinate scaling.
    
    Args:
        mask: 2D binary mask
        step: Reduction factor (how many pixels were skipped)
    
    Returns:
        Filled mask where gaps between ones are filled
    """
    # Create a copy to avoid modifying the original
    filled_mask = mask.copy()
    
    # For each row
    for i in range(mask.shape[0]):
        # Find where we have ones
        ones = np.where(mask[i] == 1)[0]
        if len(ones) > 1:
            # For each pair of ones
            for j in range(len(ones) - 1):
                # If they're close enough to be connected (within step-1 pixels)
                if ones[j+1] - ones[j] <= step:
                    # Fill all pixels between them
                    filled_mask[i, ones[j]:ones[j+1]+1] = 1
    
    # For each column
    for j in range(mask.shape[1]):
        # Find where we have ones
        ones = np.where(mask[:, j] == 1)[0]
        if len(ones) > 1:
            # For each pair of ones
            for i in range(len(ones) - 1):
                # If they're close enough to be connected (within step-1 pixels)
                if ones[i+1] - ones[i] <= step:
                    # Fill all pixels between them
                    filled_mask[ones[i]:ones[i+1]+1, j] = 1
    
    return filled_mask

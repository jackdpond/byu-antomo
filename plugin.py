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

def process_tomogram(data):
    """ Simple tomogram processing - uses contrast stretching to improve contrast. """
    print("[DEBUG] Starting process_tomogram...")
    start = time.time()
    data = data[::2, :, :]
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
        choices=["pilii", "flagellar_motor", "chemosensory_array"],
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
        
        if label in ["pilii", "flagellar_motor"]:
            name = f"{label}_points"
            if name not in viewer.layers:
                point_color = 'red' if label == 'pilii' else 'blue'
                labels_layer = viewer.add_points(name=name, ndim=3, size=4, face_color=point_color)
                labels_layer.mode = 'add'
                # Set the layer as selected for immediate editing
                viewer.layers.selection.active = labels_layer
        elif label == "chemosensory_array":
            name = f"{label}_mask"
            if name not in viewer.layers:
                image_layer = viewer.layers.selection.active
                if hasattr(image_layer, 'data'):
                    shape = image_layer.data.shape
                    mask = np.zeros(shape, dtype=np.uint8)
                    # Create labels layer
                    labels_layer = viewer.add_labels(mask, name=name)
                    # Set properties to ensure it's editable
                    labels_layer.mode = 'paint'  # Start in paint mode
                    labels_layer.selected_label = 3  # Set label to 1 (for painting)
                    labels_layer.brush_size = 20  # Set a reasonable brush size
                    # Set the layer as selected for immediate editing
                    viewer.layers.selection.active = labels_layer
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
        layer_name = f"{label}_points" if label in ["pilii", "flagellar_motor"] else f"{label}_mask"
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

        # Define column order
        columns = ["z", "y", "x", "label", "user", "timestamp", "source_path", "mask_path"]

        if label in ["pilii", "flagellar_motor"]:
            layer = viewer.layers[layer_name]
            if not layer:
                show_error(f"⚠️ No layer named {layer_name}")
                return
            coords = layer.data
            # Create DataFrame with coordinates and other columns
            df = pd.DataFrame({
                "z": coords[:, 0],
                "y": coords[:, 1],
                "x": coords[:, 2],
                "label": label,
                "user": user,
                "timestamp": timestamp,
                "source_path": source_path,
                "mask_path": None
            })
            # Ensure columns are in the correct order
            df = df[columns]

        elif label == "chemosensory_array":
            layer = viewer.layers[layer_name]
            if not layer:
                show_error(f"⚠️ No layer named {layer_name}")
                return
            mask_path = os.path.join(shared_dir, f"{label}_{user}_{timestamp}.npy")
            np.save(mask_path, layer.data)
            # Create DataFrame with all columns in correct order
            df = pd.DataFrame([{
                "z": None,
                "y": None,
                "x": None,
                "label": label,
                "user": user,
                "timestamp": timestamp,
                "source_path": source_path,
                "mask_path": mask_path
            }])
            # Ensure columns are in the correct order
            df = df[columns]

        else:
            show_error("⚠️ Unknown label type.")
            return

        all_path = os.path.join(shared_dir, "annotations_all.csv")
        label_path = os.path.join(shared_dir, f"{label}_annotations.csv")
        df.to_csv(all_path, mode='a', header=not os.path.exists(all_path), index=False)
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
        csv_path = os.path.join(shared_dir, "annotations_all.csv")
        
        if not os.path.exists(csv_path):
            stats_container.append(widgets.Label(value="No annotations found"))
            return
        
        try:
            df = pd.read_csv(csv_path)
            # Filter for current source
            df_source = df[df['source_path'] == source_path]
            
            if df_source.empty:
                stats_container.append(widgets.Label(value="No annotations for this tomogram"))
                return
            
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
                        if label in ["pilii", "flagellar_motor"]:
                            coords = np.array([[row['z'], row['y'], row['x']]])
                            layer = viewer.add_points(coords, name=display_layer_name, ndim=3, size=6, face_color='yellow')
                            layer.mode = 'pan_zoom'  # Not editable
                            layer.editable = False
                            # Jump to correct z-slice
                            z = int(row['z'])
                            jump_to_z_slice(viewer, z)
                        elif label == "chemosensory_array" and pd.notna(row['mask_path']):
                            mask = np.load(row['mask_path'])
                            layer = viewer.add_labels(mask, name=display_layer_name)
                            layer.mode = 'pan_zoom'  # Not editable
                            layer.editable = False
                            # Jump to correct z-slice (use first nonzero z)
                            z_indices = np.argwhere(mask)
                            if z_indices.size > 0:
                                z = int(z_indices[0][0])
                                jump_to_z_slice(viewer, z)
                        else:
                            show_error(f"Cannot display annotation: {label}")
                    return on_click
                btn.clicked.connect(make_on_click())
                label_container.append(btn)
                stats_container.append(label_container)
        except Exception as e:
            stats_container.append(widgets.Label(value=f"Error reading annotations: {str(e)}"))
    
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
        'tomo_ids_csv': os.path.expanduser('~/groups/fslg_imagseg/jackson/Napari/tomo_ids.csv')
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
    dialog.append(widgets.Label(value="Plugin Settings"))
    annotation_dir_edit = widgets.FileEdit(mode='d', value=plugin_config['annotation_dir'], label="Annotation Directory")
    tomo_ids_edit = widgets.FileEdit(mode='r', value=plugin_config['tomo_ids_csv'], label="Tomo IDs CSV")
    save_button = widgets.PushButton(text="Save")

    def save_settings():
        plugin_config['annotation_dir'] = annotation_dir_edit.value
        plugin_config['tomo_ids_csv'] = tomo_ids_edit.value
        save_plugin_config(plugin_config)
        # Call refresh callbacks if provided
        if refresh_callbacks:
            for cb in refresh_callbacks:
                cb()
        show_info("Plugin settings saved and widgets refreshed.")
        dialog.hide()

    save_button.clicked.connect(save_settings)
    dialog.append(annotation_dir_edit)
    dialog.append(tomo_ids_edit)
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
                'pilii', 'flagellar_motor', 'chemosensory_array'],
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
            data = mrc_to_np(file_path)
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

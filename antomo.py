# ~/napari_scripts/tomo_test.py

# Enable Qt loop (works only if launched via IPython with %gui qt)
try:
    __IPYTHON__
    get_ipython().run_line_magic("gui", "qt")
except NameError:
    pass

import napari
from plugin import create_annotation_widget, create_annotation_viewer_widget, create_tomogram_navigator_widget
import mrcfile

# with mrcfile.open("sample.rec", permissive=True) as mrc:
#     tomogram = mrc.data.copy()

# viewer = napari.view_image(tomogram, name="Tomogram", metadata={'source': "sample.rec"})
viewer = napari.Viewer()
widget = create_annotation_widget(viewer)
viewer.window.add_dock_widget(widget, area="right")

# Add annotation viewer widget
viewer_widget = create_annotation_viewer_widget(viewer)
viewer.window.add_dock_widget(viewer_widget, area="right")

# Add tomogram navigator widget
navigator_widget = create_tomogram_navigator_widget(viewer, saved_annotations_widget=viewer_widget)
viewer.window.add_dock_widget(navigator_widget, area="right")


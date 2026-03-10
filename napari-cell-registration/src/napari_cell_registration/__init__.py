"""
napari-cell-registration

DAPI-based cell segmentation, matching, and registration plugin for napari.
"""

__version__ = "0.1.0"

try:
    from ._widget import segment_cells_widget, registration_workflow_widget
    from ._manual_seg_widget import ManualSegmentationWidget
except Exception:
    # Allow importing core utilities in headless environments without napari.
    segment_cells_widget = None
    registration_workflow_widget = None
    ManualSegmentationWidget = None

__all__ = [
    "segment_cells_widget",
    "registration_workflow_widget",
    "ManualSegmentationWidget",
]

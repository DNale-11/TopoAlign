"""
Test script for napari cell registration plugin.
Quick test of the segmentation widget with existing images.
"""

import napari
from napari_cell_registration.core.io_utils import load_image

# Load test images
print("Loading test images...")
img1 = load_image("../B.tif")
img2 = load_image("../C.tif")

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")

# Create napari viewer
viewer = napari.Viewer()

# Add images as layers
viewer.add_image(img1, name="Round 1 (B.tif)")
viewer.add_image(img2, name="Round 2 (C.tif)")

print("\nPlugin loaded successfully!")
print("To use the plugin:")
print("1. Go to Plugins menu")
print("2. Select 'napari-cell-registration'")
print("3. Choose 'Cell Segmentation' for quick segmentation")
print("4. Or choose 'Registration Workflow' for complete pipeline")

# Start napari
napari.run()

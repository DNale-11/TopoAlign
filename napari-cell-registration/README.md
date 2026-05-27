# napari-cell-registration

DAPI-based cell segmentation, matching, and registration plugin for napari.

## Features

- **Cell Segmentation**: Automatic nuclear segmentation using Cellpose-SAM
- **Feature Extraction**: Extract morphological features from segmented cells
- **Cell Matching**: Match cells between two imaging rounds using feature similarity
- **Rigid Registration**: Estimate and apply rigid transforms between rounds
- **Interactive Visualization**: View and validate results directly in napari

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

1. Open napari:
   ```bash
   napari
   ```

2. Load your DAPI images as layers

3. From the Plugins menu, select:
   - **Cell Segmentation** - for quick segmentation only
   - **Registration Workflow** - for complete registration pipeline

### Large Image Segmentation

The **Cell Segmentation** widget has a `mode` option:

- `auto - chunk only if large`: use chunked segmentation only when the image is larger than `large_image_threshold_mp`.
- `full image - ignore chunk settings`: keep the previous full-image Cellpose path.
- `chunked - use label stitching`: always segment with overlapping chunks.

Use `chunk_size` to control each inference tile and `chunk_overlap` to give Cellpose context at tile borders. `stitch_labels` merges labels that touch in overlap regions; keep it enabled unless you need to inspect raw tile boundaries. Registration is unchanged.

### WSI CLI Registration

After installation, WSI mask registration can also run without napari:

```bash
wsi-mask-registration --fixed-mask round1_mask.tif --moving-mask round2_mask.tif --moving-image round2.tif --output-dir wsi_out --top-k 2000
```

Without reinstalling the package, run the same CLI as a module:

```bash
python -m napari_cell_registration.wsi_registration --fixed-mask round1_mask.tif --moving-mask round2_mask.tif --moving-image round2.tif --output-dir wsi_out --top-k 2000
```

## Requirements

- Python >= 3.10
- napari >= 0.4.18
- cellpose >= 4.0
- See `pyproject.toml` for complete dependency list

## License

BSD-3-Clause

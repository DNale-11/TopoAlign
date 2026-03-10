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

## Requirements

- Python >= 3.10
- napari >= 0.4.18
- cellpose >= 4.0
- See `pyproject.toml` for complete dependency list

## License

BSD-3-Clause

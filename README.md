# Cell Registration (DAPI-based)

Python package for nuclear segmentation, feature extraction, cell matching, and rigid alignment using Cellpose-SAM.

## Requirements
- Python 3.10 (tested)
- See `requirements.txt` for exact pins. Install with:
```bash
pip install -r requirements.txt
```

> Note: Cellpose requires PyTorch. The `torch` pin in `requirements.txt` targets CPU by default; if you want CUDA, install the matching CUDA wheel (e.g., `torch==2.3.1+cu121`) from the official PyTorch index before installing Cellpose.

## Project structure
```
cell_registration/
  __init__.py
  config.py          # global configs
  io_utils.py        # image I/O
  segmentation.py    # Cellpose-SAM wrapper
  features.py        # regionprops-based features
  matching.py        # greedy feature-space matching
  registration.py    # rigid/similarity transform
  visualization.py   # napari viewer & overlays
  main.py            # CLI demo pipeline
requirements.txt
```

## Running the demo
Run the full pipeline on two images (DAPI last channel if multichannel):
```bash
python -m cell_registration.main path/to/img1.tif path/to/img2.tif --top-k 10
```

Optional flags:
- `--napari` to open interactive viewers (requires `napari` installed).
- `--save-match-table matches.csv` to export the match pairs.
- `--save-match-overlay overlay.png` to save a side-by-side overlay with matched centroids/lines.
- `--save-segmentation-prefix outputs/segmentation` to save TIF plots of each segmentation.
- `--save-match-plot outputs/match_plot` to save a matplotlib match plot (TIF).
- `--save-registration-overlay outputs/registration_overlay` to save mask1 warped onto image2 (TIF) for visual registration check.
- `--save-features-dir outputs` to save per-cell feature tables as `round1_cells.csv` / `round2_cells.csv` (with x,y aliases for centroids).
- `--position-weight 1.0` to control spatial proximity weight in matching (0 disables position).

Example (PowerShell one-line, TIF outputs):
pretest:
python -m cell_registration.main ".\B.tif" ".\C.tif" --top-k 10 --position-weight 1 --save-segmentation-prefix .\outputs\segmentation --save-match-overlay .\outputs\match_overlay --save-match-plot .\outputs\match_plot
```powershell
python -m cell_registration.main ".\B.tif" ".\C.tif" --top-k 10 --position-weight 1 `
  --save-segmentation-prefix .\outputs\segmentation `
  --save-match-overlay .\outputs\match_overlay `
  --save-match-plot .\outputs\match_plot
```
start:
python -m cell_registration.main ".\B.tif" ".\C.tif" --top-k 10 --position-weight 1 --save-segmentation-prefix .\outputs\segmentation --save-match-overlay .\outputs\match_overlay --save-match-plot .\outputs\match_plot --save-registration-overlay .\outputs\registration_overlay --napari



The script prints Top-K matches and the estimated rotation matrix and translation vector. Overlay/CSV are written if paths are provided. Remove `--napari` if you do not need the interactive viewer.

python -m cell_registration.main ".\B.tif" ".\C.tif" --top-k 10 --position-weight 1 `
  --save-match-table .\outputs\top_matches.csv `
  --save-segmentation-prefix .\outputs\segmentation `
  --save-match-overlay .\outputs\match_overlay `
  --save-match-plot .\outputs\match_plot `
  --save-registration-overlay .\outputs\registration_overlay


python -m cell_registration.point_registration `
  .\outputs\round1_cells.csv `
  .\outputs\round2_cells.csv `
  .\outputs\top_matches.csv `
  --output-registered-csv .\outputs\round2_cells_registered.csv `
  --plot-path .\outputs\registration_plot.png `
  --napari


python -m cell_registration.main B.tif C.tif --use-spatial-clusters --n-clusters 9 --position-weight 20 --top-per-patch 6 --top-k 200 --use-ransac-transform
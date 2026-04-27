# Cell Registration (DAPI-based)

<p align="center">
  <img src="logo.png" alt="Cell Registration Logo" width="600">
</p>

A professional Python package for robust nuclear segmentation, feature extraction, and precise cellular alignment across multiplexed imaging rounds using a landmark-based registration pipeline.

## Overview

Unlike traditional patch-based or heuristic (RANSAC/ICP) approaches, this pipeline employs a **rigorous multi-stage landmark-based registration algorithm**. It aims to find highly reliable, one-to-one cellular matches based on robust local topologies before estimating the global transformation.

### Architecture Workflow

1. **Segmentation & Feature Extraction**
   - Automatically segments cells using Cellpose-SAM.
   - Extracts morphological features (`area`, `perimeter`, `eccentricity`, `shape_ratio`) and centroid coordinates. (Orientation is explicitly excluded to maintain rotation invariance).

2. **Stage 1: Loose Candidate Generation**
   - Uses KD-Tree radius queries to efficiently generate a broad set of candidate matching pairs between the fixed and moving images.

3. **Stage 2: Morphological & Positional Scoring**
   - Computes robust normalized differences (using Median and IQR/MAD) for morphological features.
   - Combines the morphological score with positional differences to filter out structurally dissimilar candidates.

4. **Stage 3: Local Topology Validation**
   - Validates the local spatial neighborhood of candidate pairs.
   - Employs scale-normalized kNN distance vectors, rotation-invariant angle histograms, and local density checks.
   - Relaxes topology constraints adaptively for border cells.

5. **Stage 4: Maximum Cardinality Min-Cost Landmark Selection**
   - Formulates the final matching as a bipartite graph problem.
   - First maximizes the total number of preserved landmark pairs (Maximum Cardinality Matching).
   - Then minimizes the overall mismatch cost to ensure strict one-to-one mapping without forcing full assignment.

6. **Stage 5: Global Transformation Estimation**
   - Utilizes all rigorously validated landmarks to estimate the final `rigid` or `similarity` transformation matrix.

## Installation

- **Python**: 3.10
- **Dependencies**: See `requirements.txt`.

```bash
pip install -r requirements.txt
```

> **Note:** Cellpose requires PyTorch. Install the appropriate CUDA wheel (e.g., `torch==2.3.1+cu121`) from the official PyTorch index before running the pipeline if you want GPU acceleration.

## Usage

Run the primary pipeline to register two images (DAPI or nuclear channel):

```bash
python -m cell_registration.main path/to/fixed.tif path/to/moving.tif
```

### Key Parameters

**General**:
- `--segmentation-only`: Only perform Cellpose segmentation and exit.
- `--napari`: Launch the interactive napari viewer for debugging and visualization.

**Algorithm Tuning**:
- `--candidate-radius-px`: Radius for initial loose spatial candidates (default: `25`).
- `--position-weight`: Weight of spatial proximity in candidate scoring.
- `--feature-score-threshold` / `--candidate-score-threshold`: Thresholds for morphology and combined position filtering.
- `--topology-radius-px` / `--topology-k`: Neighborhood radius and neighbor count for topology validation.
- `--topology-score-threshold`: Strictness of topology matching.
- `--allow-scale`: Allows similarity transform (scale + rotation + translation) instead of just rigid.

**Outputs**:
- `--save-match-table <path.csv>`: Export final landmark pairs.
- `--save-match-overlay <path.tif>`: Save a side-by-side visualization of landmark connections.
- `--save-match-plot <path.tif>`: Save a matplotlib match plot.
- `--save-registration-overlay <path.tif>`: Save the final overlay of moving image warped onto fixed image.
- `--save-features-dir <dir>`: Save extracted cell morphology tables (`round1_cells.csv` and `round2_cells.csv`).
- `--save-debug-dir <dir>`: Export detailed QA metrics (e.g., transform residuals, topology rejections).

### Example (PowerShell)

```powershell
python -m cell_registration.main ".\fixed.tif" ".\moving.tif" `
  --candidate-radius-px 25 `
  --position-weight 1.0 `
  --save-segmentation-prefix .\outputs\segmentation `
  --save-match-overlay .\outputs\match_overlay `
  --save-registration-overlay .\outputs\registration_overlay `
  --save-features-dir .\outputs
```

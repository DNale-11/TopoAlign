<div align="center">
  <img src="topoalign-logo.png" alt="TopoAlign Logo" width="600">

  # TopoAlign — Topology-Preserving Cellular Registration

  **A robust, landmark-based alignment framework for highly multiplexed tissue imaging**

  [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
  [![Cellpose](https://img.shields.io/badge/Segmentation-Cellpose--SAM-success.svg)](https://github.com/MouseLand/cellpose)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

<br/>

## 📖 Abstract

In highly multiplexed immunofluorescence (mIF) and spatial transcriptomics, iterative tissue staining and imaging often introduce non-linear deformations, mechanical drift, and sample degradation. Traditional pixel-based cross-correlation or intensity-based registration methods frequently fail under such severe structural variations. 

This repository provides a **state-of-the-art, topology-preserving cellular registration pipeline** specifically designed for challenging multiplexed datasets. By extracting deep morphological features and leveraging local cellular topology (k-NN geometry and rotation-invariant angular relationships), our method formulates cell registration as a **Maximum-Cardinality Minimum-Cost Bipartite Graph Matching** problem. This ensures mathematically rigorous, one-to-one cellular correspondence without relying on unstable heuristic approximations like RANSAC or ICP.

---

## ✨ Key Innovations

- **Topology-Guided Validation:** Computes a local spatial manifold descriptor using local density scale-normalization and rotation-invariant angular histograms, ensuring resilience to significant tissue distortion.
- **Maximum-Cardinality Landmark Matching:** Ensures strict one-to-one cellular correspondence by solving a bipartite graph matching problem—maximizing consensus landmarks while minimizing overall morphological and spatial cost.
- **Robust Morphological Descriptors:** Leverages non-parametric scaling (Median Absolute Deviation) on intrinsic cellular metrics ($Area$, $Perimeter$, $Eccentricity$, $Shape Ratio$) to establish orientation-agnostic correspondence.
- **Adaptive Boundary Relaxation:** Dynamically adjusts topological constraints for cells at tissue boundaries, mitigating the edge-effect artifacts commonly observed in spatial biology pipelines.

---

## ⚙️ Algorithmic Architecture

Our pipeline processes unaligned DAPI (or generic nuclear) channels through a rigorous 5-stage landmark validation framework:

<details open>
<summary><b>Stage 0: Deep Segmentation & Feature Extraction</b></summary>
<blockquote>
Automated nuclear segmentation via Cellpose-SAM. We extract localized, orientation-invariant morphological profiles for each segmented entity.
</blockquote>
</details>

<details open>
<summary><b>Stage 1: Spatial Candidate Generation</b></summary>
<blockquote>
A highly efficient KD-Tree spatial query establishes an initial loose bipartite graph of potential structural correspondences between consecutive imaging rounds.
</blockquote>
</details>

<details open>
<summary><b>Stage 2: Joint Morphological & Positional Scoring</b></summary>
<blockquote>
Nodes are weighted using a combined robust-normalized morphological difference function and a spatial proximity metric, systematically penalizing structurally incompatible pairs.
</blockquote>
</details>

<details open>
<summary><b>Stage 3: Local Topology Validation</b></summary>
<blockquote>
Evaluation of the local manifold. We compute a topological cost $\mathcal{L}_{topo}$ combining scale-normalized k-NN distance vectors and circular-shifted angular displacements.
</blockquote>
</details>

<details open>
<summary><b>Stage 4: Maximum-Cardinality Min-Cost Matching</b></summary>
<blockquote>
Extraction of the optimal sub-graph. The algorithm identifies the largest possible subset of strictly one-to-one matches that simultaneously adhere to the topological constraints and minimize total mismatch cost.
</blockquote>
</details>

<details open>
<summary><b>Stage 5: Global Transformation Estimation</b></summary>
<blockquote>
The highly validated, noise-free cellular landmarks are utilized to compute the optimal rigid ($SO(2)$) or similarity transformation matrix, guaranteeing globally consistent tissue alignment.
</blockquote>
</details>

---

## 🚀 Installation & Setup

### System Requirements
- **OS**: Linux / Windows
- **Python**: `3.10`

### Quick Start
Clone the repository and install dependencies:
```bash
git clone https://github.com/DNale-11/cell_registration.git
cd cell_registration
pip install -r requirements.txt
```

> **Hardware Acceleration:** For optimal Cellpose-SAM performance, install the appropriate PyTorch CUDA build (e.g., `pip install torch==2.3.1+cu121`) prior to running the pipeline.

---

## 💻 Usage & CLI Reference

The public command is `topoalign`; the legacy Python module remains available for existing scripts:

```bash
topoalign run --fixed path/to/fixed.tif --moving path/to/moving.tif \\
  --mode image --method rigid --output-dir outputs/run-001

# Legacy compatibility entry point
python -m cell_registration.main path/to/fixed.tif path/to/moving.tif
```

Running `topoalign` opens a persistent local shell. Local commands such as
`run`, `segment`, `features`, `match`, `transform`, `warp`, and `inspect` do
not require an API key. The optional Agent can be configured independently
with an API key, OpenAI-compatible base URL, and model; the shell validates
that connection before entering Agent mode.

Copy `topoalign.config.example.json` to `topoalign.config.json` to create a
local configuration. The shell reads `topoalign.config.json` from the current
project, falling back to `%USERPROFILE%\.topoalign\config.json`. The default interface language is
English; menu option `[5]` switches between English and Chinese. Registration
defaults under the JSON `registration` section are used by `run` when command
line flags do not override them. The project configuration is git-ignored
because it may contain an API key; only the credential-free example is tracked.

Inside the optional Agent, enter `/+` to list shortcuts. Available commands
include `/model`, `/model <id>`, `/api`, `/config`, `/reload`, `/tools`,
`/local`, `/result`, `/artifacts`, `/clear`, and `/exit`.

### Advanced Algorithmic Tuning

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--candidate-radius-px` | Search radius for generating initial spatial candidates ($\mathcal{N}_{radius}$) | `25` |
| `--position-weight` | Penalty coefficient for spatial displacement during morphological scoring | `1.0` |
| `--topology-radius-px` | Local neighborhood radius defining the topological manifold | `20` |
| `--topology-k` | Number of nearest neighbors evaluated for topological consensus | `5` |
| `--allow-scale` | Relaxes $SO(2)$ rigid constraint to allow scaling (Similarity transform) | `False` |

### Output Configurations

| Flag | Artifact Exported |
|------|-------------------|
| `--save-match-table` | `.csv` export of the final validated landmark pairs |
| `--save-match-overlay`| Side-by-side `.tif` visualization with topological edge connections |
| `--save-registration-overlay`| Final blended `.tif` showing the moving image warped onto the fixed reference |
| `--save-features-dir` | Directory to dump high-dimensional morphological profiles of both rounds |
| `--save-debug-dir` | QA directory containing residual distributions and topology rejection matrices |

---

## 📈 Example Execution

```powershell
python -m cell_registration.main ".\round1_dapi.tif" ".\round2_dapi.tif" `
  --candidate-radius-px 25 `
  --position-weight 1.0 `
  --topology-radius-px 20 `
  --save-segmentation-prefix .\outputs\segmentation `
  --save-match-overlay .\outputs\match_overlay `
  --save-registration-overlay .\outputs\registration_overlay `
  --save-features-dir .\outputs
```

---

<div align="center">
  <i>Developed for quantitative spatial biology and highly multiplexed imaging.</i>
</div>

<div align="center">
  <img src="topoalign-logo.png" alt="TopoAlign" width="560">

  # TopoAlign

  **Cellular image registration from CLI, Web, Agent, and napari**

  [![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![CPU](https://img.shields.io/badge/Runtime-CPU-64748B)](#cpu-installation)
  [![GPU](https://img.shields.io/badge/Runtime-NVIDIA%20GPU-76B900?logo=nvidia&logoColor=white)](#gpu-installation)
  [![Web](https://img.shields.io/badge/UI-Web-0EA5E9)](#web-usage)
  [![Agent](https://img.shields.io/badge/Agent-optional-19A974)](#agent-usage)
</div>

TopoAlign provides four ways to run the same registration workflow:

| Interface | Start command | API key required |
|---|---|---|
| CLI | `topoalign` | No |
| Web | `python webapp/backend.py` | No; only the Web Assistant needs one |
| Agent | `topoalign agent` | Yes |
| napari | `napari` | No |

The local pipeline supports image, mask, and feature-table inputs. Agent access
is optional and does not control whether local registration commands can run.

## Requirements

- Windows or Linux
- Python 3.10
- CPU mode: no NVIDIA GPU or CUDA installation required
- GPU mode: NVIDIA GPU, a compatible driver, and CUDA-enabled PyTorch
- An API key is required only for CLI Agent or Web Assistant conversations

## Clone the repository

```powershell
git clone https://github.com/DNale-11/cell_registration.git
cd cell_registration
```

Choose either the CPU installation or GPU installation below. Do not install
both PyTorch variants in the same environment.

## CPU installation

CPU mode is the simplest installation and works on machines without an NVIDIA
GPU. Segmentation can be slower on large images, but all CLI, Web, Agent, and
napari entry points remain available.

```powershell
conda create -n cell_registration_cpu python=3.10 -y
conda activate cell_registration_cpu

python -m pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cpu

pip install -r webapp/requirements.txt
pip install -e ".[agent]"
```

Verify that the environment is using CPU PyTorch:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected result:

```text
CUDA available: False
```

### CPU usage

Use `--no-gpu` when running image segmentation from the CLI:

```powershell
topoalign run `
  --fixed fixed.tif `
  --moving moving.tif `
  --mode image `
  --method rigid `
  --no-gpu `
  --output-dir outputs/cpu-run
```

For a standalone segmentation stage:

```powershell
topoalign segment `
  --image fixed.tif `
  --no-gpu `
  --output-dir outputs/cpu-segmentation
```

Start the Web interface manually in the CPU environment and leave **Use GPU**
disabled:

```powershell
conda activate cell_registration_cpu
python webapp/backend.py
```

## GPU installation

GPU mode is recommended for image segmentation and large-image workflows.
The example below installs the official CUDA 12.8 PyTorch wheels. If the local
NVIDIA driver requires another build, select the appropriate command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/) and keep
the project version constraints in `requirements.txt`.

```powershell
conda create -n cell_registration_gpu python=3.10 -y
conda activate cell_registration_gpu

python -m pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 `
  --index-url https://download.pytorch.org/whl/cu128

pip install -r webapp/requirements.txt
pip install -e ".[agent]"
```

Verify CUDA before starting a registration:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

`available` must be `True`. If it is `False`, do not enable GPU in TopoAlign
until the PyTorch/driver installation is corrected.

### GPU usage

Enable GPU segmentation from the CLI:

```powershell
topoalign run `
  --fixed fixed.tif `
  --moving moving.tif `
  --mode image `
  --method rigid `
  --gpu `
  --output-dir outputs/gpu-run
```

For a standalone segmentation stage:

```powershell
topoalign segment `
  --image fixed.tif `
  --gpu `
  --output-dir outputs/gpu-segmentation
```

On Windows, `webapp/start_server.bat` activates the
`cell_registration_gpu` environment and checks whether CUDA is available:

```powershell
.\webapp\start_server.bat
```

GPU mode primarily accelerates segmentation and GPU-enabled plugin operations.
Feature extraction, matching, transform estimation, file I/O, and some WSI
operations may still use the CPU.

## Verify TopoAlign

The following checks apply to both installations:

```powershell
topoalign --version
topoalign --help
```

If `topoalign` is not recognized, activate the correct environment and run:

```powershell
pip install -e ".[agent]"
```

## CLI usage

Run without a subcommand to enter the persistent interface:

```powershell
topoalign
```

The main interface lets you enter the optional Agent, configure the Agent API,
view local command help, change the interface language, or run a local command
directly:

```text
TopoAlign> inspect fixed.tif
TopoAlign> run --fixed fixed.tif --moving moving.tif --mode image --method rigid --output-dir outputs/run-001
```

### Complete registration

```powershell
topoalign run `
  --fixed fixed.tif `
  --moving moving.tif `
  --mode image `
  --method rigid `
  --registration-channel -1 `
  --output-dir outputs/run-001
```

Available transform methods are `rigid`, `similarity`, and `affine`.

Use existing masks without segmentation:

```powershell
topoalign run `
  --fixed-mask fixed_mask.tif `
  --moving-mask moving_mask.tif `
  --mode mask `
  --method similarity `
  --output-dir outputs/mask-run
```

Use existing feature tables:

```powershell
topoalign run `
  --fixed-features fixed_features.csv `
  --moving-features moving_features.csv `
  --mode features `
  --method affine `
  --output-dir outputs/feature-run
```

Mask and feature workflows do not need PyTorch or GPU computation during the
matching and transform stages.

### Individual stages

```powershell
# Segment
topoalign segment --image fixed.tif --output-dir outputs/segmentation

# Extract features
topoalign features --mask fixed_mask.tif --output-dir outputs/features

# Match features
topoalign match `
  --fixed-features fixed_features.csv `
  --moving-features moving_features.csv `
  --fixed-shape 2048 2048 `
  --output-dir outputs/matching

# Estimate the moving-to-fixed transform
topoalign transform `
  --fixed-features fixed_features.csv `
  --moving-features moving_features.csv `
  --matches matches.csv `
  --method rigid `
  --output-dir outputs/transform

# Warp the moving image
topoalign warp `
  --moving moving.tif `
  --transform transform.moving_to_fixed.json `
  --fixed-shape 2048 2048 `
  --output-dir outputs/warp

# Inspect an input or result
topoalign inspect outputs/run-001/result.json
```

Use `topoalign <command> --help` for all parameters. Add `--json` to supported
commands when a machine-readable stdout result is needed.

## Web usage

The local Web application provides:

- fixed and moving image upload and preview;
- one-fixed-to-many-moving batch registration;
- segmentation-only execution;
- normal and WSI registration modes;
- CPU/GPU selection;
- fixed, moving, mask, registered, overlay, and match-line views;
- progress updates, result downloads, and system resource statistics;
- an optional AI Assistant that can inspect Web state and call allowlisted
  TopoAlign actions.

Start the server from the repository root:

```powershell
# CPU
conda activate cell_registration_cpu
python webapp/backend.py

# GPU
conda activate cell_registration_gpu
python webapp/backend.py
```

Open [http://localhost:8000](http://localhost:8000) in a browser.

### Web workflow

1. Upload one fixed image.
2. Upload one or multiple moving images.
3. Select Cellpose or CellViT when available.
4. Enable **Use GPU** only in a verified GPU environment.
5. Select **Normal registration** or **WSI registration**.
6. Adjust parameters or keep the defaults.
7. Run segmentation only, or start the full registration batch.
8. Switch between moving-image results and inspect masks, overlay, and match
   lines.
9. Download the registered moving image or other generated artifacts.

Uploads and task results are written under `data/`. This directory is ignored
by Git and should not be committed.

### Web Assistant

Open **AI Configuration** in the Web interface and enter:

- an OpenAI-compatible API URL;
- an API key;
- a model ID supported by that API account.

The Web application can run without these values. They are required only for
the Assistant panel. The Assistant can inspect uploaded images, update Web
parameters, start an analysis, inspect completed results, and summarize a
one-fixed-to-many-moving batch through allowlisted tools.

The Web server listens on `0.0.0.0:8000`. It is intended for trusted local or
private-network use. Add authentication and a restricted CORS policy before
exposing it to an untrusted network.

## Agent usage

The CLI Agent interprets natural-language tasks and calls a restricted set of
TopoAlign tools. It can inspect inputs and runtime resources, generate and
validate configuration, run individual stages or complete registration,
process one fixed image against multiple moving images, and diagnose
structured results.

It cannot execute arbitrary shell commands, modify source code, or access
files outside the selected workspace.

### Configure the Agent

Run the main interface and select the Agent configuration option:

```powershell
topoalign
```

Enter the API URL, API key, and model ID. TopoAlign makes a minimal model call
to verify that the complete combination works.

Configuration can also be loaded from JSON:

```powershell
Copy-Item topoalign.config.example.json topoalign.config.json
```

```json
{
  "agent": {
    "enabled": true,
    "base_url": "https://api.openai.com/v1",
    "api_key": "",
    "api_key_env": "OPENAI_API_KEY",
    "model": "YOUR_MODEL_ID",
    "timeout_seconds": 30,
    "max_tool_rounds": 8
  }
}
```

To keep the key outside JSON:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
topoalign agent
```

### Agent examples

Start the conversational interface:

```powershell
topoalign agent
```

CPU request:

```text
You> Inspect the runtime and both images. Use CPU segmentation, run rigid registration, and diagnose the result.
```

GPU request:

```text
You> Confirm that CUDA is available. Use GPU segmentation, register fixed.tif to moving.tif, and inspect the output quality.
```

Batch request:

```text
You> Inspect fixed.tif and every image in moving/, choose explicit parameters, run one-fixed-to-many-moving registration into outputs/batch-01, and diagnose every result.
```

One-shot usage:

```powershell
topoalign agent --prompt "Inspect the runtime and explain whether CPU or GPU mode is available"
```

The Agent shows `Thinking...` while a model request or local tool is still
running. This is an activity indicator, not private model reasoning.

### Agent commands

Enter `/+` inside the Agent to show the command list.

| Command | Action |
|---|---|
| `/model` | Show the current model |
| `/model <id>` | Validate and switch model |
| `/api` | Test the configured URL, key, and model |
| `/config` | Show active configuration with the key redacted |
| `/reload` | Reload the local JSON configuration |
| `/tools` | List allowlisted Agent tools |
| `/local <command>` | Run a local TopoAlign command |
| `/result` | Read the latest `result.json` |
| `/artifacts` | List the latest output artifacts |
| `/clear` | Clear model conversation context |
| `/exit` | Return to the TopoAlign main interface |

The project-local `.agents/skills/topoalign-analysis` instructions are shared
by the CLI Agent and Web Assistant.

## JSON registration configuration

The credential-free template is `topoalign.config.example.json`. Copy it to
`topoalign.config.json`, then edit local values:

```powershell
Copy-Item topoalign.config.example.json topoalign.config.json
topoalign run --config topoalign.config.json
```

Important sections:

```text
cli
agent
registration
  segmentation
  matching
  transform
  output
```

Set `registration.segmentation.gpu` to `false` for CPU or `true` for GPU.
Command-line parameters override JSON values. `topoalign.config.json` is
ignored by Git because it may contain credentials and local paths.

## Output files

A complete image run can produce:

```text
outputs/run-001/
|-- config.resolved.json
|-- result.json
|-- diagnostics.json
|-- fixed_features.csv
|-- moving_features.csv
|-- matches.csv
|-- transform.moving_to_fixed.json
|-- registered_moving.tif
|-- valid_overlap_mask.tif
`-- overlay.tif
```

The exact files depend on the input mode and output settings.
`registered_moving.tif` contains only the warped moving image; the visualization
is stored separately as `overlay.tif`. Transform direction is always named
`moving_to_fixed` in structured outputs.

## napari usage

The root requirements install napari. Install the plugin package and launch:

```powershell
pip install -e .\napari-cell-registration
napari
```

Open the TopoAlign/cell-registration widgets from napari's **Plugins** menu.
Choose the GPU option only when `torch.cuda.is_available()` is `True`.

## Legacy entry point

The original Python namespace remains available for existing scripts:

```powershell
python -m cell_registration.main fixed.tif moving.tif
```

The public application and command name is `TopoAlign`.

## Troubleshooting

### `topoalign` is not recognized

```powershell
conda activate cell_registration_cpu  # or cell_registration_gpu
pip install -e ".[agent]"
```

### GPU is enabled but CUDA is unavailable

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If the final value is `False`, reinstall a PyTorch wheel compatible with the
local NVIDIA driver or continue with `--no-gpu`.

### Web server dependencies are missing

```powershell
pip install -r webapp/requirements.txt
```

### Agent model cannot be called

Use `/api` to test the complete URL/key/model combination and `/model <id>` to
validate a model before switching. A provider's public model catalog does not
prove that the current API key can call every listed model.

### Run without an API key

Use CLI, Web registration, or napari directly. Only Agent conversations need
an API key:

```powershell
topoalign run --fixed fixed.tif --moving moving.tif --no-gpu --output-dir outputs/local-run
```

<div align="center">
  <img src="topoalign-logo.png" alt="TopoAlign" width="560">

  # TopoAlign

  **Command-line and napari tools for cellular image registration**

  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![CLI](https://img.shields.io/badge/CLI-topoalign-1496D4)](#command-line-usage)
  [![Agent](https://img.shields.io/badge/Agent-optional-19A974)](#agent-usage)
</div>

TopoAlign can be used from the command line, from its interactive Agent, or
through the napari plugin. The local registration commands do not require an
API key.

<details>
<summary><strong>中文说明</strong></summary>

TopoAlign 提供命令行、交互式 Agent 和 napari 插件三种使用方式。本地分割、
特征提取、匹配、变换估计和图像重采样均不需要 API key；只有使用 Agent
理解指令并操控这些功能时才需要配置模型 API。

</details>

## Requirements

- Windows or Linux
- Python 3.10 or newer
- A CUDA-compatible environment is optional
- An API key is optional and is only required for Agent mode

## Installation

Clone the repository and create an isolated environment:

```powershell
git clone https://github.com/DNale-11/cell_registration.git
cd cell_registration
conda create -n topoalign python=3.10 -y
conda activate topoalign
```

Install the registration dependencies and TopoAlign in editable mode:

```powershell
pip install -r requirements.txt
pip install -e .
```

Install the optional Agent dependency when conversational control is needed:

```powershell
pip install -e ".[agent]"
```

Verify the installation:

```powershell
topoalign --version
topoalign --help
```

## Start the interactive CLI

Run TopoAlign without a subcommand:

```powershell
topoalign
```

The persistent shell provides four main actions:

1. Enter the optional Agent.
2. Configure and test the Agent API connection.
3. View local CLI command help.
4. Exit.

Use menu option `5` to switch between English and Chinese. Local commands can
also be entered directly at the `TopoAlign>` prompt, for example:

```text
TopoAlign> inspect fixed.tif
TopoAlign> run --fixed fixed.tif --moving moving.tif --mode image --method rigid --output-dir outputs/run-001
```

## Command-line usage

### Complete registration

Register two images:

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

Register existing label masks without running segmentation:

```powershell
topoalign run `
  --fixed-mask fixed_mask.tif `
  --moving-mask moving_mask.tif `
  --mode mask `
  --method similarity `
  --output-dir outputs/mask-run
```

Start from existing feature tables:

```powershell
topoalign run `
  --fixed-features fixed_features.csv `
  --moving-features moving_features.csv `
  --mode features `
  --method affine `
  --output-dir outputs/feature-run
```

Use `--json` to print the structured result to stdout.

### Run individual stages

Every registration stage can be run independently:

```powershell
# Segment an image
topoalign segment --image fixed.tif --output-dir outputs/fixed-segmentation

# Extract features from a label mask
topoalign features --mask fixed_mask.tif --output-dir outputs/fixed-features

# Match two feature tables
topoalign match `
  --fixed-features fixed_features.csv `
  --moving-features moving_features.csv `
  --fixed-shape 2048 2048 `
  --output-dir outputs/matching

# Estimate a moving-to-fixed transform
topoalign transform `
  --fixed-features fixed_features.csv `
  --moving-features moving_features.csv `
  --matches matches.csv `
  --method rigid `
  --output-dir outputs/transform

# Warp an image into fixed coordinates
topoalign warp `
  --moving moving.tif `
  --transform transform.moving_to_fixed.json `
  --fixed-shape 2048 2048 `
  --output-dir outputs/warp

# Inspect an input, config, transform, or result manifest
topoalign inspect outputs/run-001/result.json
```

Use `topoalign <command> --help` for the complete options of a stage.

## JSON configuration

Copy the credential-free example before editing local settings:

```powershell
Copy-Item topoalign.config.example.json topoalign.config.json
```

On Linux:

```bash
cp topoalign.config.example.json topoalign.config.json
```

The configuration contains these sections:

```text
cli
agent
registration
  segmentation
  matching
  transform
  output
```

Run registration from JSON:

```powershell
topoalign run --config topoalign.config.json
```

Command-line values override values from the JSON file. The local
`topoalign.config.json` file is ignored by Git because it may contain an API
key. Commit only `topoalign.config.example.json`, which contains no
credentials.

## Agent usage

The Agent converts natural-language requests into calls to TopoAlign's
allowlisted registration tools. It does not provide a web server and it does
not execute arbitrary shell commands.

### Configure the Agent from the CLI

Start the persistent interface:

```powershell
topoalign
```

Select **Configure/test Agent API URL, API key, and model**. Enter:

- API base URL, such as `https://api.openai.com/v1`
- API key
- Model ID supported by that API account

TopoAlign makes a minimal model request and reports whether the selected model
can be called. The API key is visible while it is entered, as requested by the
CLI design, so avoid sharing terminal screenshots.

The same values can be entered in the local JSON file:

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

For better credential isolation, leave `api_key` empty and use an environment
variable:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
topoalign
```

An OpenAI-compatible provider can be used by changing `base_url` and `model`
to values supported by that provider.

### Talk to the Agent

Choose **Enter optional Agent** from the main shell, or start it directly:

```powershell
topoalign agent
```

Example conversation:

```text
You> inspect fixed.tif and moving.tif, then run rigid registration into outputs/sample-01

Agent> Thinking...

Agent> The registration completed. The moving-to-fixed transform and result manifest were written to outputs/sample-01.
```

The `Thinking...` animation indicates that the model or a registration tool is
still running. It is an activity indicator, not a display of private model
reasoning.

Run a single Agent request without entering the REPL:

```powershell
topoalign agent --prompt "Inspect the project and explain the rigid registration inputs"
```

### Agent commands

Enter `/+` inside the Agent to display the command list.

| Command | Action |
|---|---|
| `/model` | Show the current model |
| `/model <id>` | Validate and switch to a model |
| `/api` | Test the configured API URL, key, and model |
| `/config` | Show active configuration with the key redacted |
| `/reload` | Reload `topoalign.config.json` |
| `/tools` | List the allowlisted Agent tools |
| `/local <command>` | Run a local TopoAlign command |
| `/result` | Read the latest `result.json` |
| `/artifacts` | List outputs from the latest run |
| `/clear` | Clear the current model conversation context |
| `/exit` | Return to the TopoAlign main shell |

The Agent can inspect project files, validate inputs and configuration, run
registration stages, and read structured results. Source-code modification,
arbitrary shell execution, and access outside the selected workspace are not
enabled.

## Output files

A complete image run writes structured artifacts such as the following to the
selected output directory (the exact files depend on the supplied inputs and
output settings):

```text
outputs/run-001/
├── config.resolved.json
├── result.json
├── diagnostics.json
├── fixed_features.csv
├── moving_features.csv
├── matches.csv
├── transform.moving_to_fixed.json
├── registered_moving.tif
├── valid_overlap_mask.tif
└── overlay.tif
```

`registered_moving.tif` contains only the warped moving image.
`overlay.tif` is stored separately. The transform direction is always named
`moving_to_fixed` in output manifests.

## napari plugin

Install the plugin from the repository:

```powershell
pip install -e .\napari-cell-registration
napari
```

Open the TopoAlign/cell-registration widgets from napari's **Plugins** menu.
The napari interface and CLI expose the same registration workflow through
different user interfaces.

## Legacy entry point

Existing scripts can continue using the original package namespace:

```powershell
python -m cell_registration.main fixed.tif moving.tif
```

The public application and command name is `TopoAlign`; the
`cell_registration` Python namespace remains available for compatibility.

## Troubleshooting

### `topoalign` is not recognized

Activate the environment where TopoAlign was installed and reinstall the
editable package:

```powershell
conda activate topoalign
pip install -e .
```

### Agent dependency is missing

```powershell
pip install -e ".[agent]"
```

### Agent model cannot be called

Use `/api` to test the complete URL/key/model combination. Use
`/model <id>` to validate a specific model before switching. TopoAlign does not
treat a provider's public model catalog as proof that the current API key can
call every model.

### Local registration without an API key

Use any local command directly. Agent configuration is optional:

```powershell
topoalign run --fixed fixed.tif --moving moving.tif --output-dir outputs/local-run
```

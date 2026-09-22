<div align="center">
  <img src="docs/_static/topoalign-release-logo.png" alt="TopoAlign" width="560">

  # TopoAlign

  **Topology-guided cellular image registration**

  [![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](docs/installation.rst)
  [![Documentation](https://img.shields.io/badge/docs-User%20Manual-8CA1AF?logo=readthedocs&logoColor=white)](docs/README.md)
</div>

TopoAlign registers cellular microscopy images using cell morphology and local
spatial relationships. It accepts images, instance masks, and feature tables,
with rigid, similarity, and affine transforms in the CLI. The public source
also includes a napari plugin and an optional conversational Agent.

**Fixed** defines the reference coordinates. **Moving** is transformed into
Fixed space; CLI transform files record the direction as `moving_to_fixed`.

## User manual

Read the [English User Manual](docs/README.md) for the complete documentation.

| Task | Guide |
|---|---|
| Install the core CLI or CPU/GPU image environment | [Installation](docs/installation.rst) |
| Try a small reproducible example | [Quick start](docs/quickstart.rst) |
| Run single pairs, individual stages, or scripted batches | [CLI guide](docs/cli_guide.rst) |
| Inspect images and edit masks interactively | [napari guide](docs/napari_guide.rst) |
| Configure optional API-assisted operation | [Agent guide](docs/agent_guide.rst) |
| Choose parameters and assess alignment | [Parameters](docs/parameters.rst) and [outputs](docs/outputs.rst) |
| Resolve common problems | [Troubleshooting](docs/troubleshooting.rst) |

The manual describes public version 0.1.0. CLI and napari pipelines have
different defaults and export conventions. The documented source revision
does not include a Web server; see [Web interface availability](docs/web_guide.rst).

## Quick installation check

This small mask example needs no GPU, segmentation model, or API key.

```shell
git clone https://github.com/DNale-11/TopoAlign.git
cd TopoAlign
conda create -n topoalign_core python=3.10 -y
conda activate topoalign_core
python -m pip install --upgrade pip
python -m pip install -e . "matplotlib>=3.7,<3.10"
python -m pip check
topoalign --version
python docs/examples/create_demo_masks.py
topoalign run --fixed-mask demo/fixed_mask.tif --moving-mask demo/moving_mask.tif --mode mask --method rigid --output-dir outputs/demo-mask
topoalign inspect outputs/demo-mask/result.json
```

The example should recover 12 matches and a Moving-to-Fixed translation of
`x = -6`, `y = +4` pixels. It is a synthetic installation check, not a
biological accuracy benchmark. Use a new output directory for each run.

For raw-image segmentation, use the manual's
[CPU or GPU installation](docs/installation.rst). The published requirements
constrain PyTorch to `>=2.2,<2.5`; follow the matching PyTorch/torchvision
versions in that guide.

## Documentation development

The manual uses Sphinx and supports Read the Docs. See
[build and publishing instructions](docs/README.md#build-the-html-manual).
Documentation builds use a separate Python 3.11 environment.

Report reproducible problems through
[GitHub Issues](https://github.com/DNale-11/TopoAlign/issues).

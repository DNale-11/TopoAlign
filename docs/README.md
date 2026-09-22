# TopoAlign User Manual

This English manual covers the public TopoAlign 0.1.0 source at commit
`693c6781e162`. Start with installation and the synthetic mask example, then
choose the CLI, napari, or optional Agent workflow.

## Read the manual on GitHub

- [Overview](overview.rst): workflow, inputs, coordinates, and interface selection.
- [Installation](installation.rst): core CLI, CPU/GPU segmentation, napari, and Agent setup.
- [Quick start](quickstart.rst): a reproducible mask example and your first image pair.
- [CLI guide](cli_guide.rst): complete runs, individual stages, data schemas, and batches.
- [napari guide](napari_guide.rst): widgets, manual segmentation, registration, and WSI.
- [Agent guide](agent_guide.rst): API setup, conversation commands, and local tool use.
- [Parameters](parameters.rst): defaults, configuration files, and tuning.
- [Outputs and quality control](outputs.rst): generated files, coordinates, and diagnostics.
- [Troubleshooting](troubleshooting.rst): symptoms, recovery steps, and known limitations.
- [Web interface availability](web_guide.rst): why earlier Web commands are absent from this revision.

The `.rst` pages are the source of the Sphinx manual. GitHub displays their
content, but Sphinx-specific cross-references and download directives work best
in the HTML build. The links above provide navigation directly on GitHub.

## Build the HTML manual

Use a separate Python 3.11 environment for documentation. Building the manual
does not require installing TopoAlign, PyTorch, Cellpose, or napari.

```shell
conda create -n topoalign_docs python=3.11 -y
conda activate topoalign_docs
python -m pip install -r docs/requirements.txt
python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`, or preview through a local server:

```shell
python -m http.server 8080 --bind 127.0.0.1 --directory docs/_build/html
```

Then visit [the local manual](http://127.0.0.1:8080).

Check external links separately, with network access:

```shell
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
```

Inspect `docs/_build/linkcheck/output.txt` when an external service blocks or
times out. A successful HTML build checks internal references and assets; it
does not establish external site availability.

## Publish through Read the Docs

1. Commit the `docs/` source and root `.readthedocs.yaml` to
   [DNale-11/TopoAlign](https://github.com/DNale-11/TopoAlign).
2. In Read the Docs, import that GitHub repository, or update the repository URL
   of the existing project after the repository rename.
3. Set the project language to English and its default branch to `main`.
4. Use the project slug `topoalign` if it is available and belongs to this
   project. The intended URL is `https://topoalign.readthedocs.io/en/latest/`.
5. Trigger a build. The existing `.readthedocs.yaml` installs only
   `docs/requirements.txt` and treats Sphinx warnings as errors.
6. Verify navigation, downloads, search, and the **Edit on GitHub** link in the
   deployed site before promoting the public documentation URL in the README.

If another slug is used, replace the intended documentation URL accordingly.
A local build does not publish or update a Read the Docs project.

## Maintain the manual

- Check examples against the published revision, including the exact CLI
  options and output names. Update the revision statement when behavior changes.
- Keep CLI and napari defaults and export semantics distinct.
- Run the synthetic quick start after changing any command in it.
- Add each new user page to `index.rst` and to the GitHub navigation above.
- Keep private data, API keys, model weights, and generated analysis results
  outside documentation source. Do not commit `docs/_build/`.

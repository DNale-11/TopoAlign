# TopoAlign documentation

The online manual is built with Sphinx and the Read the Docs theme.

## Build locally

```powershell
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build succeeds.

## Publish on Read the Docs

1. Sign in to Read the Docs with GitHub.
2. Import `DNale-11/cell_registration`.
3. Use the project slug `topoalign` so the public URL matches the README badge.
4. Trigger the first build. `.readthedocs.yaml` is detected automatically.

If a different slug is used, update the Documentation link in the root `README.md`.

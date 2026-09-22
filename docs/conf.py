from __future__ import annotations

project = "TopoAlign"
author = "TopoAlign contributors"
copyright = "2026, TopoAlign contributors"
version = "0.1"
release = "0.1.0"

extensions: list[str] = []
templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"

html_theme = "sphinx_rtd_theme"
html_title = "TopoAlign User Manual"
html_logo = "_static/topoalign-release-logo.png"
html_favicon = "_static/topoalign-release-logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "style_external_links": True,
}
html_context = {
    "display_github": True,
    "github_user": "DNale-11",
    "github_repo": "TopoAlign",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

linkcheck_timeout = 20
linkcheck_retries = 2

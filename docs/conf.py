from __future__ import annotations

import subprocess
from pathlib import Path

project = "Stashed Bloom Filter"
author = "COMP0252 Group Project"
copyright = "2026, COMP0252 Group Project"

extensions = [
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"

DOCS_DIR = Path(__file__).resolve().parent
DOXYFILE = DOCS_DIR / "Doxyfile"
DOXYGEN_XML_DIR = DOCS_DIR / "_build" / "doxygen" / "xml"

breathe_projects = {"stashed_bloom_filter": str(DOXYGEN_XML_DIR)}
breathe_default_project = "stashed_bloom_filter"


def run_doxygen() -> None:
    try:
        subprocess.run(["doxygen", str(DOXYFILE)], cwd=DOCS_DIR, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("doxygen is required to build docs. Install doxygen and retry.") from exc


def setup(app):
    app.connect("builder-inited", lambda _: run_doxygen())

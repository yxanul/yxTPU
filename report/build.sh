#!/bin/bash
# Rebuild the technical report: figures (when the data exports are
# present), then the PDF via tectonic.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f figures/data/60sndjih.csv ]; then
  (cd figures && uv run --with matplotlib --with pandas python make_figures.py)
fi
tectonic main.tex
echo "built $(pwd)/main.pdf"

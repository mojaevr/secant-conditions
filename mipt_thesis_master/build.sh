#!/usr/bin/env bash
# Сборка main.pdf под TeX Live 2026 basic.
# Дважды pdflatex — чтобы устаканились \tableofcontents и \ref/\cite.
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/texlive/2026basic/bin/universal-darwin:$PATH
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "OK: main.pdf готов ($(wc -c < main.pdf) байт)"

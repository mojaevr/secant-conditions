#!/usr/bin/env bash
# Сборка main.pdf под TeX Live 2026 basic.
# Цикл pdflatex → bibtex → pdflatex → pdflatex нужен:
#   * первый pdflatex создаёт main.aux со списком \cite-ключей;
#   * bibtex генерирует main.bbl по стилю ugost2003s (пакет gost,
#     установлен в ~/Library/texmf через `tlmgr --usermode install gost`);
#   * второй pdflatex подхватывает main.bbl;
#   * третий pdflatex устаканивает \tableofcontents и перекрёстные ссылки.
set -e
cd "$(dirname "$0")"
export PATH=/usr/local/texlive/2026basic/bin/universal-darwin:$PATH
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "OK: main.pdf готов ($(wc -c < main.pdf) байт)"

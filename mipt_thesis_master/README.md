# SP-Broyden — магистерская диссертация

«Исследование глобальных квазиньютоновских методов оптимизации для задач
математического программирования и вариационных неравенств».

Автор: Можаев Р. М. (М05-404а), научный руководитель: Камзолов Д. И.,
Физтех-школа прикладной математики и информатики МФТИ, кафедра проблем
передачи информации и анализа данных.

Базовая схема VIJI-Restarted взята из соавторской публикации
Agafonov, Ostroukhov, Mozhaev et al., NeurIPS 2024
([arXiv:2405.15990](https://arxiv.org/abs/2405.15990)); в той работе
использовалось иное приближение якобиана. Настоящая диссертация
переносит схему VIJI-Restarted на SP-Broyden и приводит соответствующий
теоретический анализ и численные эксперименты.


## Сборка

```sh
./build.sh
```

Скрипт прогоняет `pdflatex -interaction=nonstopmode main.tex` дважды (первый
проход — заполняет `.aux`/`.toc`, второй — устаканивает `\ref`/`\cite`).
Используется TeX Live 2026 basic; путь к бинарникам зашит как
`/usr/local/texlive/2026basic/bin/universal-darwin` — при необходимости
поправьте `build.sh` под свой дистрибутив.

Сборка библиографии встроена в источник (ручной `\begin{thebibliography}`
в `sections/bibliography.tex`, не biblatex), отдельный прогон `biber` не
требуется.


## Структура исходников

```
mipt_thesis_master/
├── main.tex                       — корневой файл (преамбула + \include всего)
├── mipt-thesis-bs.cls             — класс шаблона МФТИ (бакалаврский, переиспользован)
├── mipt-thesis-biblatex.sty       — сопровождающий стилевик (biblatex не используется)
├── build.sh                       — двукратный pdflatex
├── check_static.py                — статическая сверка cite/ref/label
├── README.md                      — этот файл
│
├── sections/
│   ├── abstract_ru.tex            — аннотация (рус)
│   ├── abstract_en.tex            — abstract (en)
│   ├── notation.tex               — список обозначений
│   ├── introduction.tex           — введение (актуальность, цель, новизна)
│   ├── literature_review.tex      — обзор литературы
│   ├── ch1_jacobian.tex           — гл. 1: общая формула обновления якобиана
│   ├── ch2_sp_broyden.tex         — гл. 2: SP-Broyden, теоремы, эксперименты, L-SP
│   ├── ch3_symmetric.tex          — гл. 3: SS-SR1, scaling, сравнение QN-методов
│   ├── ch4_sp_afd.tex             — гл. 4: VIJI-Restarted и SP-AFD
│   ├── conclusion.tex             — заключение
│   ├── reproducibility.tex        — программная реализация и воспроизводимость
│   └── bibliography.tex           — ручной \begin{thebibliography} (48 записей)
│
└── appendix/
    ├── app_alignment.tex          — Лемма 4.3: выравнивание VIJI-итераций
    └── app_sp_afd_proof.tex       — Теорема 4.8: полное доказательство SP-AFD
```

В корне репозитория (`../`) лежат вычислительные ноутбуки, диагностические
скрипты и сырые результаты; см. ниже.


## Соответствие глав, ноутбуков и скриптов

| Глава / раздел | Ноутбук | Скрипт диагностики | Сырые данные | Картинки |
|---|---|---|---|---|
| гл. 2 «SP-Broyden»: эксперименты сходимости | `tezisy.ipynb` | `diag_sp_broyden.py` | — | `fig_sp_broyden_conv.pdf`, `fig_sp_broyden_jacerr.pdf`, `fig_sp_broyden_pvar.pdf`, `fig_sp_broyden_cond.pdf` |
| гл. 2, раздел `sec:highdim` (L-SP-Broyden, $n\le 10^4$) | — | `diag_highdim.py` | `highdim_results.npz` | `fig_highdim_conv.pdf`, `fig_highdim_summary.pdf`, `fig_highdim_pvar.pdf` |
| гл. 3, статистика SS-конструкции в $n\in\{10,20,50\}$ — пара SR1/SS-SR1 и пара PSB/SS-PSB, по 50 случайных стартов; convergence-фигуры для обеих пар + накопленная невязка прошлых секущих $\Vert B_k S_p - Y_p\Vert_F$ для PSB-пары (тh. ss_props п. 3) | — | `diag_ndim_stat.py` | `ndim_stat.npz`, `ndim_stat_summary.txt` | `fig_ndim_stat_sr1.pdf`, `fig_ndim_stat_psb.pdf`, `fig_ndim_stat_rpast_psb.pdf` |
| гл. 4 «VIJI-Restarted»: основные эксперименты | `pics.ipynb` | `run_seeds.py`, `diag_sp_afd.py` | `results.npz` | `fig_viji_conv.pdf`, `fig_viji_seeds.pdf`, `fig_sp_afd_gap_T.pdf`, `fig_sp_afd_problems.pdf`, `fig_sp_afd_cond14.pdf`, `fig_sp_afd_rstar.pdf` |
| гл. 4, baseline Anderson($m,\beta$) | — | `diag_anderson.py` | `anderson_baseline.npz` | `fig_anderson_baseline.pdf`, `fig_anderson_summary.pdf` |

Каждый PDF-плот лежит в `mipt_thesis_master/` (рядом с `.tex`-исходниками)
для прямой подстановки через `\includegraphics`. Сырые `*.npz` — в корне
репозитория, чтобы не раздувать поддиректорию диссертации.


## Воспроизводимость

Подробности — в `sections/reproducibility.tex` (включается в финальный PDF).
Краткий справочник:

- Python 3.10, NumPy 1.26, SciPy 1.11, Matplotlib 3.8 (TeX Live 2025+).
- Seed'ы — через `numpy.random.default_rng(seed)` (генератор PCG64),
  глобальный `np.random.seed` не используется.
- Единый критерий остановки: $\varepsilon_{\mathrm{tol}}=10^{-10}$,
  норма зависит от типа задачи (см. таблицу в `reproducibility.tex`).
- 10 seed'ов на конфигурацию для VIJI-Restarted (`run_seeds.py`),
  10–50 случайных направлений на точку для радиальных диагностик в гл. 3
  (`diag_table31_ci.py`, `diag_ss_sr1_scaling.py`).

Запуск любого диагностического скрипта из корня репозитория:

```sh
cd ..
python diag_sp_broyden.py        # ⇒ fig_sp_broyden_*.pdf в mipt_thesis_master/
python diag_highdim.py           # ⇒ fig_highdim_*.pdf, highdim_results.npz
# и т. п.
```


## Статическая проверка перед сборкой

В `mipt_thesis_master/` лежит `check_static.py` — без TeX-сборки сверяет
`\cite{...}` ↔ `\bibitem{...}`, `\ref{...}/\eqref{...}` ↔ `\label{...}`,
ловит дубликаты `\label`. На текущий момент:

- 48/48 `\cite` ↔ `\bibitem` (с учётом `\cite[opt]{key}`),
- 0 потерянных `\ref`/`\eqref`,
- 0 дубликатов `\label`.

```sh
cd mipt_thesis_master
python check_static.py
```


## Заметки по сборке (TeX Live 2026)

В преамбуле `main.tex` стоят целевые workaround'ы:

- `\RequirePackage{etoolbox}` до `\documentclass` — обходит протекание
  `\globaldefs=1` в LaTeX3-хеши при загрузке класса.
- Объявление `\c@v@normal`/`\c@v@bold`/`\c@v@italic`/`\c@v@bolditalic`
  через `\newcount` до загрузки `hyperref` — обходит `pd1enc.def:38`,
  где `\advance` срабатывает на `\relax`.
- `\usepackage{lmodern}` — заменяет CMR на Latin Modern с полной
  поддержкой T2A/T1/TS1/PD1/PU и убирает font shape warning'и.
- `\setlength{\headheight}{15pt}` — убирает предупреждение `fancyhdr`.

`mathtools` не подключён: его v1.31 в TL 2026 падает на собственной
строке `\EQ_MakeRobust\MT_extended_eqref:n`. Достаточно
`amsmath`/`amssymb`, которые класс уже подгружает.


## Лицензия и происхождение шаблона

Класс `mipt-thesis-bs` (и сопровождающие `mipt-thesis*.cls`/`.sty`/`.bbx`)
— шаблон МФТИ для бакалаврских/магистерских дипломов авторства
А. А. Киселёва и М. С. Долгоносова (см. `LICENSE`). Здесь использован
с минимальными правками (TL 2026 совместимость, `\onehalfspacing`,
устранение `\globaldefs=1`).

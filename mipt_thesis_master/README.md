# SP-Broyden — магистерская диссертация

«Исследование глобальных квазиньютоновских методов оптимизации для задач
математического программирования и вариационных неравенств».

Автор: Можаев Р. М. (М05-404а), научный руководитель: Камзолов Д. И.,
Физтех-школа прикладной математики и информатики МФТИ, кафедра проблем
передачи информации и анализа данных.


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
└── sections/
    ├── abstract_ru.tex            — аннотация
    ├── notation.tex               — список обозначений
    ├── introduction.tex           — введение (актуальность, цель, новизна)
    ├── ch1_jacobian.tex           — гл. 1: общая формула обновления якобиана
    ├── ch2_sp_broyden.tex         — гл. 2: SP-Broyden, теоремы, эксперименты, L-SP
    ├── ch3_symmetric.tex          — гл. 3: SS-U, SS-SR1, SS-PSB, статистика
    ├── conclusion.tex             — заключение
    └── bibliography.tex           — ручной \begin{thebibliography}
```

В корне репозитория (`../`) лежат вычислительные ноутбуки, диагностические
скрипты и сырые результаты; см. ниже.


## Соответствие глав, ноутбуков и скриптов

| Глава / раздел | Ноутбук | Скрипт диагностики | Сырые данные | Картинки |
|---|---|---|---|---|
| гл. 2 «SP-Broyden»: эволюция $\|B_k - J(x_k)\|_F$ | `tezisy.ipynb` | `diag_jacerr_stat.py` | — | `fig_sp_broyden_jacerr.pdf` |
| гл. 2, раздел `sec:highdim` (L-SP-Broyden, $n\le 10^4$) | — | `diag_highdim_stat.py` | `highdim_results.npz` | `fig_highdim_conv.pdf`, `fig_highdim_summary.pdf`, `fig_highdim_pvar.pdf` |
| гл. 3, статистика SS-конструкции в $n\in\{10,20,50\}$ — пара SR1/SS-SR1 и пара PSB/SS-PSB, по 50 случайных стартов; convergence-фигуры + накопленная невязка прошлых секущих $\Vert B_k S_p - Y_p\Vert_F$ для PSB-пары (теорема `thm:ss_props` п. 3) | — | `diag_ndim_stat.py` | `ndim_stat.npz`, `ndim_stat_summary.txt` | `fig_ndim_stat_sr1.pdf`, `fig_ndim_stat_psb.pdf`, `fig_ndim_stat_rpast_psb.pdf` |

Каждый PDF-плот лежит в `mipt_thesis_master/` (рядом с `.tex`-исходниками)
для прямой подстановки через `\includegraphics`. Сырые `*.npz` — в корне
репозитория, чтобы не раздувать поддиректорию диссертации.


## Воспроизводимость

- Python 3.10, NumPy 1.26, SciPy 1.11, Matplotlib 3.8 (TeX Live 2025+).
- Seed'ы — через `numpy.random.default_rng(seed)` (генератор PCG64),
  глобальный `np.random.seed` не используется.
- Единый критерий остановки: $\varepsilon_{\mathrm{tol}}=10^{-10}$
  (норма зависит от типа задачи).
- Для статистических диагностик (гл. 3, paired-design) — 50 случайных
  направлений на точку с фиксированным seed.

Запуск любого диагностического скрипта из корня репозитория:

```sh
cd ..
python diag_jacerr_stat.py       # ⇒ fig_sp_broyden_jacerr.pdf в mipt_thesis_master/
python diag_highdim_stat.py      # ⇒ fig_highdim_*.pdf, highdim_results.npz
python diag_ndim_stat.py         # ⇒ fig_ndim_stat_*.pdf, ndim_stat.npz
```


## Статическая проверка перед сборкой

В `mipt_thesis_master/` лежит `check_static.py` — без TeX-сборки сверяет
`\cite{...}` ↔ `\bibitem{...}`, `\ref{...}/\eqref{...}` ↔ `\label{...}`,
ловит дубликаты `\label`. На текущий момент:

- все `\cite` имеют `\bibitem` (с учётом `\cite[opt]{key}`),
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

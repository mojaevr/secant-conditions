# SP-Broyden

Магистерская диссертация: **«Исследование глобальных квазиньютоновских методов
оптимизации для задач математического программирования»**.

Автор: Можаев Р. М., МФТИ, ФПМИ, кафедра ППИАД, группа М05-404а.
Научный руководитель: Камзолов Д. И.

## Что в работе

Диссертация развивает количественный локальный анализ **мультисекущего
обновления Бройдена** (Schnabel, 1983) для нелинейных систем
и строит его секантно-стабилизированный симметричный аналог для
гладкой оптимизации.

Основные результаты:

- **Глава 2 (Projected Broyden, PB).** Оценка bounded deterioration
  мультисекущего обновления усилена с неравенства Гэя–Шнабеля
  до точного *равенства* с проектором ранга $p{+}1$ и явной
  не зависящей от $n$ константой
  $C_{\mathrm{PB}} = L(1{+}C_s/2)\sqrt{p+1}\,\kappa_p$.
  Установлена $O(\rho_k)$-аппроксимация якобиана на подпространстве
  прошлых шагов. Limited-memory вариант **L-PB** разобран отдельно.
- **Глава 3 (Symmetric SP-Broyden, SS-U).** Ранг-1 коррекция
  в $s_k^{\perp}$, сохраняющая текущее секущее условие и симметрию,
  с условной Q-сверхлинейной сходимостью по Dennis–Moré
  и отдельной леммой о bounded deterioration композитного обновления.
  Частный случай SS-SR1 эмпирически демонстрирует расширенную область
  сходимости по сравнению с классическим SR1.

Основная публикация, в которую входит схема VIJI-Restarted:
Agafonov A., Ostroukhov P., Mozhaev R. et al.
*Exploring Jacobian inexactness in second-order methods for variational
inequalities: lower bounds, quasi-Newton updates, and restarts.*
NeurIPS 2024. arXiv:2405.15990.

## Структура репозитория

```
SP-Broyden/
├── mipt_thesis_master/         # LaTeX-источник диссертации
│   ├── main.tex                # главный файл сборки
│   ├── main.pdf                # готовый PDF
│   ├── build.sh                # двойной прогон pdflatex
│   ├── sections/               # аннотация, обозначения, введение, 3 главы, заключение
│   │   ├── abstract_ru.tex
│   │   ├── notation.tex
│   │   ├── introduction.tex
│   │   ├── ch1_jacobian.tex    # обзор приближений якобиана
│   │   ├── ch2_sp_broyden.tex  # Projected Broyden + L-PB
│   │   ├── ch3_symmetric.tex   # Symmetric SP-Broyden (SS-U / SS-SR1)
│   │   ├── conclusion.tex
│   │   ├── bibliography.tex
│   │   └── references.bib
│   ├── mipt-thesis-bs.cls      # шаблон МФТИ
│   └── fig_*.pdf               # рисунки для глав 2–3
├── diag_highdim.py             # L-PB-реализации (библиотека для diag_highdim_stat)
├── diag_highdim_stat.py        # fig_highdim_conv: статистика n=10^4/10^5
├── diag_jacerr_stat.py         # fig_sp_broyden_jacerr: ошибка якобиана PB
├── diag_ndim_stat.py           # fig_ndim_stat_*: SS-SR1/SS-PSB vs SR1/PSB
├── diag_ndim_noarmijo.py       # run_no_armijo (библиотека для diag_ss_local_basin)
├── diag_ss_local_basin.py      # fig_ss_local_basin: бассейн локальной сходимости
├── diag_ss_local_quadratic.py  # fig_ss_local_quadratic: квадратичная сходимость
├── diag_ss_local_sweep.py      # supplementary: sweep по (n, κ, R)
├── ndim_stat_summary.txt       # сводка для диссертационных fig_ndim_stat_*
├── *.npz                       # сырые результаты экспериментов
├── (2019) Положение о ВКР.pdf  # нормативный документ МФТИ
└── README.md
```

## Сборка PDF

`build.sh` собирает диссертацию через `pdflatex`, на macOS патчит
путь к TeX Live 2026basic, если `pdflatex` нет в `$PATH`.

```bash
bash mipt_thesis_master/build.sh
```

Итог — `mipt_thesis_master/main.pdf`.

## Воспроизведение экспериментов

Скрипты `diag_*.py` требуют Python 3.10+ и NumPy/SciPy/matplotlib.
Каждый скрипт автономен: запуск без аргументов перестраивает
соответствующие `.npz`/`*_summary.txt` и обновляет PDF-рисунок
в `mipt_thesis_master/`.

```bash
pip install numpy scipy matplotlib
python diag_ndim_stat.py        # пример: пересобрать fig_ndim_stat_*
```

## Лицензия

Шаблон диссертации (`mipt_thesis_master/mipt-thesis-*.cls`,
`mipt_thesis_master/LICENSE`) — под лицензией авторов шаблона МФТИ.
Остальной материал (тексты глав, скрипты экспериментов) —
© Можаев Р. М., 2026.

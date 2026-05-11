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
├── diag_global_study.py        # глобальный study (fig_global_*)
├── diag_highdim.py             # L-PB на высокоразмерных задачах
├── diag_highdim_stat.py        # статистика по высокоразмерным запускам
├── diag_jacerr_stat.py         # ошибка якобиана (fig_sp_broyden_jacerr)
├── diag_ndim_stat.py           # SS-SR1/SS-PSB vs SR1/PSB (fig_ndim_stat_*)
├── diag_ndim_noarmijo.py       # SS без Armijo (глобальный режим)
├── diag_ndim_noarmijo_local.py # SS без Armijo (локальный режим)
├── diag_ss_local_basin.py      # бассейн локальной сходимости (fig_ss_local_basin)
├── diag_ss_local_quadratic.py  # квадратичная сходимость (fig_ss_local_quadratic)
├── diag_ss_local_sweep.py      # вспомогательный sweep
├── *_summary.txt, *.npz        # результаты экспериментов (входы для рисунков)
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

# experiments/legacy/

Архив скриптов, которые **не порождают рисунки, попавшие в финальные главы**
диссертации SP-Broyden. Не удаляем — могут понадобиться для ревью или
повторного запуска под другую гипотезу.

## Что здесь и почему

| Скрипт | Что производил | Почему в архиве |
|---|---|---|
| `diag_sp_broyden.py` | `fig_sp_broyden_jacerr.pdf` (single-seed), `fig_sp_broyden_pvar.pdf`, `fig_sp_broyden_cond.pdf` | Single-seed версия `jacerr` заменена на `diag_jacerr_stat.py` (медиана + IQR по seed-ам); `pvar`/`cond` не попали в главы. |
| `diag_randomized_sketch.py` | `fig_random_sketch_decay/rate.pdf` | Опровергнутая гипотеза о случайной проекции (см. `lessons_ch2_rewrite.md`). |
| `diag_pre_asymptotic.py` | `fig_pre_asymptotic_{dm,nu,alpha,kpre}.pdf` | Доасимптотический режим не вошёл в главу 2. |
| `diag_hybrid_random.py` | `fig_hybrid_random.pdf` | Гибрид со случайным направлением — отброшен. |
| `diag_basin.py` | `fig_spb_basin.pdf` | Бассейн SP-Broyden не используется; для главы 3 (SS-SR1) бассейн строится из `basin.ipynb`. |
| `diag_linear_finite.py` | `fig_linear_finite{,_rank}.pdf` | Линейная+конечноразностная аппроксимация — не попала в финал. |
| `diag_sp_ablation.py` | `fig_sp_ablation.pdf` | Аблейшен SP — отброшен после ревью гл. 2. |
| `diag_block_restart.py` | `fig_block_restart.pdf` | Блочные рестарты — отброшенная ветвь. |
| `diag_ss_sr1.py` | `fig_ss_sr1_ndim_{conv,rpast,summary}.pdf` (single-trajectory) | Заменён на `diag_ndim_stat.py` (50 случайных стартов, paired-design); single-trajectory рисунки убраны как методологически слабые (одна «удачная» траектория не отражает доли сходимости). |
| `diag_qn_compare.py` | `fig_qn_compare_{conv,summary,rpast}.pdf` (single-trajectory, 6 методов) | То же: статистика по 50 стартам теперь в `diag_ndim_stat.py` (фигура `fig_ndim_stat_qn.pdf`). |

## Если потребуется вернуть

```bash
git mv experiments/legacy/<имя>.py .
```

Все скрипты сохранены в git-истории — `git log --follow` покажет, как они
менялись до архивации.

# `dissertation/`

LaTeX source of the master's thesis:

> **Deep Reinforcement Learning for Competitive Agents in MicroRTS: Architecture, Training, and Tournament Evaluation**
> Mathis Delsart, UCLouvain (EPL), 2026.

The compiled PDF is shipped at [`dissertation.pdf`](dissertation.pdf).

## Layout

| Path | What it is |
|---|---|
| [`dissertation.tex`](dissertation.tex) | Top-level document. `\include`s every chapter. |
| [`preamble.tex`](preamble.tex) | Packages, hyperref setup, custom macros. |
| [`dissertation.cls`](dissertation.cls) | Class file (EPL master-thesis derivative). |
| [`config/`](config/) | Per-package configuration (geometry, fonts, fancyhdr). |
| [`content/`](content/) | Front/back matter content fragments shared across chapters. |
| [`covers/`](covers/) | Front + back cover PDFs (EPL master-thesis template). |
| [`covers-template/`](covers-template/) | Editable LaTeX source of the cover (regenerates `covers/_frontbanner_*.pdf`). |
| [`chapters/`](chapters/) | One subfolder per chapter (front matter + 14 chapters + appendix). |
| [`figs/figs-tex/`](figs/figs-tex/) | Standalone LaTeX `.tex` figures (TikZ, pgfplots). |
| [`figs/figs-python/`](figs/figs-python/) | Python scripts that regenerate PDF plots from `data/`. |
| [`figs/figs-pdf/`](figs/figs-pdf/) | 30 generated PDFs (the compiled figures used by the thesis). |
| [`fonts/`](fonts/) | Proprietary fonts used by the thesis (Cambria Math, Tahoma, Bookman Old Style, Courier New). |
| [`poster/`](poster/) | Defence poster PDF. |
| [`references.bib`](references.bib) | Biblatex bibliography. |

## Chapter ordering

```
00_dedication/   00_abstract/   00_acks/   00_interlude/
01_introduction/
02_microrts_decision_problem/
03_reinforcement_learning/
04_classical_approaches_rts/
05_deep_rl_approaches_rts/
06_evaluation_framework/
07_env_stack/
08_architectures/
09_training_system/
10_results/
11_discussion/
12_limitations/
13_future_work/
14_conclusion/
appendix/
```

## How to compile

```bash
cd dissertation/
latexmk -pdf -interaction=nonstopmode dissertation.tex
```

`latexmk` resolves the bibliography (biblatex/biber), the glossary
(`makeglossaries`) and re-runs LaTeX until cross-references stabilise.
Build artefacts (`.aux`, `.bbl`, `.fdb_latexmk`, ...) live next to the
source in this folder for convenience.

## Regenerating the Python figures

The data-driven plots in [`figs/figs-pdf/`](figs/figs-pdf/) are produced
by the scripts in [`figs/figs-python/`](figs/figs-python/), which read
from the shipped `data/` tree:

```bash
# From the repo root, after a successful microrts_agent install:
python dissertation/figs/figs-python/bc_vs_scratch_overall.py
python dissertation/figs/figs-python/best_vs_stier_100M.py
python dissertation/figs/figs-python/finetune_schedule.py
python dissertation/figs/figs-python/generalization_probes.py
python dissertation/figs/figs-python/metrics_comparison.py
python dissertation/figs/figs-python/phased_schedule.py
python dissertation/figs/figs-python/rush_collapse.py
python dissertation/figs/figs-python/rush_fragility.py
```

Shared helpers live in [`figs/figs-python/_data.py`](figs/figs-python/_data.py)
(path resolution + log/CSV parsing) and
[`figs/figs-python/_style.py`](figs/figs-python/_style.py) (palette +
mpl style). Each script writes its PDF straight into
[`figs/figs-pdf/`](figs/figs-pdf/), overwriting the shipped one.

The path-resolution helper `find_run_dir(name)` resolves a run name by
trying, in order:
`data/agents/<name>/` -> `data/ablation/arch/agent/<name>/` ->
`data/ablation/feat/agent/<name>/` -> `outputs/runs/<name>/`. So the
scripts re-run on a fresh clone without any local
`outputs/runs/` from a previous training.

Figures that depend on training metrics resolve through the run names
shipped under [`../data/agents/`](../data/agents/) (e.g.
`UECD-SingleMap-Best`, `UECD-MultiMap`, `UECD-BC-PPO`).

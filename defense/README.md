# `defense/`

Beamer slide deck for the master's thesis oral defense:

> **Deep Reinforcement Learning for Competitive Agents in MicroRTS: Architecture, Training, and Tournament Evaluation**
> Mathis Delsart, UCLouvain (EPL), 2026.

A 20-minute talk in ten parts, from RTS background and the research
questions through the UECD architecture, the training recipe, the
tournament results, and the discussion. The compiled deck is shipped at
[`slides/main.pdf`](slides/main.pdf).

## Layout

| Path | What it is |
|---|---|
| [`slides/main.tex`](slides/main.tex) | The deck. Single-file `metropolis` Beamer document (`aspectratio=169`, pdflatex). |
| [`slides/main.pdf`](slides/main.pdf) | The compiled slides, for direct viewing. |
| [`figures/`](figures/) | Figures used by the deck: recolored `*_bg.png` diagrams, cropped `map_*.png` thumbnails, and the UCLouvain/EPL logo chip. |
| [`recolor_figures.py`](recolor_figures.py) | Regenerates the `*_bg.png` set from the dissertation figures. |
| [`make_feat_ablation_chart.py`](make_feat_ablation_chart.py) | Regenerates the feature-ablation chart (`figures/feat_ablation_full.png`). |

## How to compile

```bash
cd defense/slides/
latexmk -pdf -interaction=nonstopmode main.tex
```

Needs a TeX distribution with the `metropolis` theme, `tcolorbox`,
`colortbl`, and `fontawesome5`. All raster figures referenced by the
deck are committed under [`figures/`](figures/), so the source compiles
on a fresh clone without re-running the figure scripts below.

## Regenerating the figures

Every chart and diagram has its white background recolored to the slide
canvas (`#FAFAFA`) so figures blend into the slides with no white box.
[`recolor_figures.py`](recolor_figures.py) renders the dissertation
figures at high resolution and maps near-white pixels to the canvas
colour, writing each `<name>_bg.png` into [`figures/`](figures/). The
feature-ablation bar chart is generated separately:

```bash
# From the repo root:
python defense/recolor_figures.py
python defense/make_feat_ablation_chart.py
```

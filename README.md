<div align="center">

# Deep Reinforcement Learning for Competitive Agents in MicroRTS

### Architecture, Training, and Tournament Evaluation

**[Mathis Delsart](https://orcid.org/0009-0005-1136-9203)** · Master's thesis · [UCLouvain](https://uclouvain.be/) · 2026

[![CI](https://github.com/mathisdelsart/microrts-drl-uecd/actions/workflows/ci.yml/badge.svg)](https://github.com/mathisdelsart/microrts-drl-uecd/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/mathisdelsart/microrts-drl-uecd?label=release&color=success)](https://github.com/mathisdelsart/microrts-drl-uecd/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/LICENSE)
[![Master's Thesis](https://img.shields.io/badge/Master's%20Thesis-UCLouvain-9cf.svg)](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/dissertation/dissertation.pdf)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-green.svg)](https://arxiv.org/abs/1707.06347)
[![Built on MicroRTS](https://img.shields.io/badge/Built%20on-MicroRTS-orange.svg)](https://github.com/santiontanon/microrts)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

<table align="center">
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/mathisdelsart/microrts-drl-uecd-website/main/videos/UECD-Best_vs_RAISocketAI_P0.gif" width="100%" alt="UECD-Best vs RAISocketAI"></td>
<td width="50%"><img src="https://raw.githubusercontent.com/mathisdelsart/microrts-drl-uecd-website/main/videos/UECD-Best_vs_CoacAI_P0.gif" width="100%" alt="UECD-Best vs CoacAI"></td>
</tr>
<tr>
<td align="center"><b>UECD-Best vs RAISocketAI</b> (IEEE-CoG competition winner)</td>
<td align="center"><b>UECD-Best vs CoacAI</b></td>
</tr>
</table>

<div align="center">

> A deep-RL agent for MicroRTS fusing a U-Net spatial encoder with an entity-level Transformer (**UECD**). Trained on a **9.47-GPU-day academic budget**, it tops a 19-agent IEEE-CoG-style tournament at **96.67%** pool win rate and beats the reigning competition winner **RAISocketAI** in **65.7%** of head-to-head games. Fully released and installable from `pip install -e .`.

</div>

---

## About

Real-time strategy (RTS) games are among the most demanding benchmarks for
sequential decision-making: players gather resources, coordinate many units,
and plan over long horizons under real-time and combinatorial-action
constraints. AlphaStar reached Grandmaster level in StarCraft II at the cost
of hundreds of accelerators running for weeks, beyond academic reach;
**MicroRTS** distills these difficulties onto small grid maps while keeping
training tractable on a modest budget, and has been the subject of an annual
competition since 2017.

This master's thesis investigates deep reinforcement learning for MicroRTS,
guided by **two questions**: *which architectural and algorithmic design
decisions most improve a MicroRTS agent*, and *whether one competitive with
the strongest prior competition entries can be trained within an academic
compute budget*. Starting from the Gym-microRTS GridNet baseline and taking
**RAISocketAI** (the first DRL winner of the competition) as reference,
every design decision is ablated individually before being combined.

**Contributions:**
- A reproducible **CoG-style tournament framework** over twelve maps and
  fifteen reference agents under five ranking metrics.
- An extended, modular **Java/Python environment stack** with composable
  wrappers and vectorized self-play.
- The **UECD architecture** fusing multi-scale convolution, entity-level
  Transformer reasoning, and bottleneck self-attention.
- A **modular PPO pipeline** whose mechanisms are ablated individually.
- A formal analysis of a **discount-induced reward collapse** under
  shaped-to-sparse annealing.

**Results.** The resulting agent, **UECD-Best**, combines these under a
two-phase opponent-curriculum fine-tuning schedule. On `basesWorkers16x16A`
it tops a 19-agent round-robin tournament (**96.67% win rate**, first on
four of five metrics) and wins **65.7% of its head-to-head games against
RAISocketAI**, using **9.47 GPU-days** and about **350M steps**, below the
**23.6 GPU-days and 500M steps** RAISocketAI reports for its small-map
subset. A second agent, **UECD-MultiMap**, trained across five layouts of
three different sizes, spreads competence evenly with no per-map collapse,
showing that the padded environment and a prioritized-level-replay
curriculum make cross-layout training feasible.

The open-source pipeline released with this thesis offers a DRL substrate
for future generalist agents and hybrid DRL/LLM systems.

---

## Read more

- **Dissertation** (full text): [`dissertation/dissertation.pdf`](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/dissertation/dissertation.pdf)
- **Short paper** (CoG 2026, under review): [`cog-2026-paper/paper.pdf`](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/cog-2026-paper/paper.pdf)
- **Supplementary site** (tournament visualisations, game recordings, analyses): <https://mathisdelsart.github.io/microrts-drl-uecd-website/>

---

## Released and installable

Frozen at **v0.1.0** and archived on Zenodo. The full pipeline (training,
evaluation, tournament, behaviour cloning, benchmarks, analysis) is
reproducible from a single `pip install -e ".[dev,tournament]"`. No
proprietary dependencies. Two shell scripts under
[`setup/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/setup)
fully automate the install on either a laptop or a CECI HPC cluster
(modules, Java 17 JDK, JNI bridge build, RAISocketAI competition wheel,
Python 3.6 sidecar for the UTS_Imass bot). Once installed, the **unified
CLI** dispatches every operation:

```bash
microrts-agent --help
# train | evaluate | tournament | bc | bench | analysis
```

Seven numbered notebooks under
[`examples/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/examples)
walk through each subcommand with shipped agents and tiny smoke budgets,
so you can verify the install and learn the CLI in ~30 minutes total.

---

## Repository tour

| Folder | What's there | Doc |
|---|---|---|
| [`microrts_agent/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/microrts_agent) | The importable Python package: 80 modules across `architectures/`, `training/`, `envs/`, `wrappers/`, `tournament/`, `registries/`, `bc/`, `bench/`, `analysis/`, plus the vendored MicroRTS engine + JNI bridge | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/microrts_agent/README.md) |
| [`data/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/data) | Curated artefacts shipped with the repo: 9 trained agents, BC teacher dataset, headline tournaments (50 PDFs), 85 ablation runs, 36 game recordings, generalisation probes | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/data/README.md) |
| [`dissertation/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/dissertation) | LaTeX source of the master's thesis, including 30 PDF figures regenerated by Python scripts in `figs/figs-python/` from the shipped `data/` tree | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/dissertation/README.md) |
| [`cog-2026-paper/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/cog-2026-paper) | CoG 2026 short-paper submission (IEEEtran, single-file LaTeX) | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/cog-2026-paper/README.md) |
| [`experiments/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/experiments) | 19 SLURM batch scripts: every shipped agent and ablation is reproducible from these drivers on a CECI HPC node | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/experiments/README.md) |
| [`examples/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/examples) | 7 numbered Jupyter notebooks, one per CLI subcommand, from `00_navigate` (install + sanity check) to `06_analysis` (metrics + audit + parameter counts) | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/examples/README.md) |
| [`tests/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/tests) | pytest smoke suite: 127 tests covering imports, every CLI subcommand, every shipped agent, the JNI bridge, end-to-end train/evaluate, and dataset schemas (~90s in CI) | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/tests/README.md) |
| [`setup/`](https://github.com/mathisdelsart/microrts-drl-uecd/tree/v0.1.0/setup) | Install scripts: `local.sh` (conda env on laptop), `cluster.sh` (venv on CECI HPC), `_common.sh` (shared helpers) | [README](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/setup/README.md) |

---

## Acknowledgments

Every experiment ran on the HPC clusters of the **CÉCI** (Consortium des
Équipements de Calcul Intensif). See
[`ACKNOWLEDGMENTS.md`](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/ACKNOWLEDGMENTS.md)
for the full acknowledgment.

## License

Released under the [MIT License](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/LICENSE) © 2026 Mathis Delsart.

## Citation

If you use this code or any of the shipped artefacts in academic work,
please cite the project ([`CITATION.cff`](https://github.com/mathisdelsart/microrts-drl-uecd/blob/v0.1.0/CITATION.cff) is the source of truth):

```bibtex
@software{delsart_microrts_drl_uecd_2026,
  author  = {Delsart, Mathis},
  title   = {{Deep Reinforcement Learning for Competitive Agents in MicroRTS: Architecture, Training, and Tournament Evaluation}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/mathisdelsart/microrts-drl-uecd/releases/tag/v0.1.0},
  note    = {Master's thesis, UCLouvain},
}
```

## Author

**[Mathis Delsart](https://orcid.org/0009-0005-1136-9203)**, Master's thesis, UCLouvain.

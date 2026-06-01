<img width="100%" alt="MicroRTS DRL Agent" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0208,50:4a0d2b,100:0d0208&height=220&section=header&text=Deep%20RL%20for%20Competitive%20MicroRTS%20Agents&fontSize=38&fontColor=fff&animation=fadeIn&fontAlignY=42&desc=Architecture,%20Training,%20and%20Tournament%20Evaluation&descSize=18&descAlignY=56" />

<div align="center">

**[Mathis Delsart](https://github.com/mathisdelsart)** · Master's thesis · [UCLouvain](https://uclouvain.be/) · 2026 · [DOI: 10.5281/zenodo.20481385](https://doi.org/10.5281/zenodo.20481385)

<p>
  <a href="https://github.com/mathisdelsart/microrts-drl-uecd/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/mathisdelsart/microrts-drl-uecd/ci.yml?branch=main&style=for-the-badge&label=CI&color=0d0208&labelColor=4a0d2b" alt="CI" /></a>
  <a href="https://github.com/mathisdelsart/microrts-drl-uecd/releases"><img src="https://img.shields.io/badge/release-v0.1.0-0d0208?style=for-the-badge&labelColor=4a0d2b" alt="Release" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0d0208?style=for-the-badge&labelColor=4a0d2b" alt="License: MIT" /></a>
  <a href="https://uclouvain.be/"><img src="https://img.shields.io/badge/Master's%20Thesis-UCLouvain-0d0208?style=for-the-badge&labelColor=4a0d2b" alt="Master's Thesis: UCLouvain" /></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-0d0208?style=for-the-badge&logo=python&logoColor=e05080&labelColor=4a0d2b" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-0d0208?style=for-the-badge&logo=pytorch&logoColor=e05080&labelColor=4a0d2b" alt="PyTorch 2.0+" />
  <a href="https://arxiv.org/abs/1707.06347"><img src="https://img.shields.io/badge/Algorithm-PPO-0d0208?style=for-the-badge&labelColor=4a0d2b" alt="Algorithm: PPO" /></a>
  <a href="https://github.com/santiontanon/microrts"><img src="https://img.shields.io/badge/Built%20on-MicroRTS-0d0208?style=for-the-badge&labelColor=4a0d2b" alt="Built on MicroRTS" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=for-the-badge" alt="Ruff" /></a>
</p>

</div>

<table align="center">
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/mathisdelsart/microrts-drl-uecd-website/main/videos/UECD-Best_vs_RAISocketAI_P0.gif" width="100%" alt="UECD-Best vs RAISocketAI" /></td>
<td width="50%"><img src="https://raw.githubusercontent.com/mathisdelsart/microrts-drl-uecd-website/main/videos/UECD-Best_vs_CoacAI_P0.gif" width="100%" alt="UECD-Best vs CoacAI" /></td>
</tr>
<tr>
<td align="center"><b>UECD-Best vs RAISocketAI</b></td>
<td align="center"><b>UECD-Best vs CoacAI</b></td>
</tr>
</table>

> A **DRL agent** for MicroRTS fusing a U-Net spatial encoder with an entity-level Transformer (**UECD**). Trained on a **9.47-GPU-day** academic budget, it tops a 19-agent IEEE-CoG-style tournament at **96.67%** pool win rate and beats the reigning competition winner **RAISocketAI** in **65.7%** of head-to-head games.
>
> Fully released and installable from `pip install -e .`.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d0208,50:4a0d2b,100:0d0208&height=2" width="100%" alt="" />

## About

Real-time strategy (RTS) games are among the most demanding benchmarks for
sequential decision-making: players gather resources, coordinate many units,
and plan over long horizons under real-time and combinatorial-action
constraints. AlphaStar reached Grandmaster level in StarCraft II at the cost
of hundreds of accelerators running for weeks, beyond academic reach;
**MicroRTS** distills these difficulties onto small grid maps while keeping
training tractable on a modest budget, and has been the subject of an annual
competition since 2017.

This master's thesis investigates DRL for MicroRTS, guided by **two
questions**: *which architectural and algorithmic design decisions most
improve a MicroRTS agent*, and *whether one competitive with the strongest
prior competition entries can be trained within an academic compute budget*.
Starting from the Gym-microRTS GridNet baseline and taking **RAISocketAI**
(the first DRL winner of the competition) as reference, every design
decision is ablated individually before being combined.

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

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d0208,50:4a0d2b,100:0d0208&height=2" width="100%" alt="" />

## Read more

- **Dissertation** (full text): [`dissertation/dissertation.pdf`](dissertation/dissertation.pdf)
- **Short paper** (CoG 2026, under review): [`cog-2026-paper/paper.pdf`](cog-2026-paper/paper.pdf)
- **Supplementary site** (tournament visualisations, game recordings, analyses): <https://mathisdelsart.github.io/microrts-drl-uecd-website/>

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d0208,50:4a0d2b,100:0d0208&height=2" width="100%" alt="" />

## Released and installable

Frozen at **v0.1.0** and archived on Zenodo. The full pipeline (training,
evaluation, tournament, behaviour cloning, benchmarks, analysis) is
reproducible from one of two automated setup scripts under
[`setup/`](setup/). Each script checks for a Java 17 JDK, rebuilds the
JNI bridge from source, installs the Python package with all extras
(`[dev,tournament]`), and fetches the RAISocketAI competition wheel (the
cluster path also bootstraps a Python 3.6 sidecar for the UTS_Imass
bot). No proprietary dependencies.

```bash
# Laptop (macOS or Linux): creates the `microrts_agent` conda env
bash setup/local.sh
conda activate microrts_agent

# Or HPC cluster (CECI: Lyra, Manneback, ...): creates ./cluster_venv
bash setup/cluster.sh
source cluster_venv/bin/activate
```

Once activated, the **unified CLI** dispatches every operation:

```bash
microrts-agent --help
# train | evaluate | tournament | bc | bench | analysis
```

Seven numbered notebooks under [`examples/`](examples/) walk through each
subcommand with shipped agents and tiny smoke budgets, so you can verify
the install and learn the CLI in ~30 minutes total.

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d0208,50:4a0d2b,100:0d0208&height=2" width="100%" alt="" />

## Repository tour

| Folder | What's there | Doc |
|---|---|---|
| [`microrts_agent/`](microrts_agent/) | The importable Python package: 80 modules across `architectures/`, `training/`, `envs/`, `wrappers/`, `tournament/`, `registries/`, `bc/`, `bench/`, `analysis/`, plus the vendored MicroRTS engine + JNI bridge | [README](microrts_agent/README.md) |
| [`data/`](data/) | Curated artefacts shipped with the repo: 9 trained agents, BC teacher dataset, headline tournaments (50 PDFs), 85 ablation runs, 36 game recordings, generalisation probes | [README](data/README.md) |
| [`dissertation/`](dissertation/) | LaTeX source of the master's thesis, including 30 PDF figures regenerated by Python scripts in `figs/figs-python/` from the shipped `data/` tree | [README](dissertation/README.md) |
| [`cog-2026-paper/`](cog-2026-paper/) | CoG 2026 short-paper submission (IEEEtran, single-file LaTeX) | [README](cog-2026-paper/README.md) |
| [`experiments/`](experiments/) | 19 SLURM batch scripts: every shipped agent and ablation is reproducible from these drivers on a CECI HPC node | [README](experiments/README.md) |
| [`examples/`](examples/) | 7 numbered Jupyter notebooks, one per CLI subcommand, from `00_navigate` (install + sanity check) to `06_analysis` (metrics + audit + parameter counts) | [README](examples/README.md) |
| [`tests/`](tests/) | pytest smoke suite: 124 tests covering imports, every CLI subcommand, every shipped agent, the JNI bridge, end-to-end train/evaluate, and dataset schemas (~90s in CI) | [README](tests/README.md) |
| [`setup/`](setup/) | Install scripts: `local.sh` (conda env on laptop), `cluster.sh` (venv on CECI HPC), `_common.sh` (shared helpers) | [README](setup/README.md) |

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0d0208,50:4a0d2b,100:0d0208&height=2" width="100%" alt="" />

## Acknowledgments

Every experiment ran on the HPC clusters of the **CÉCI** (Consortium des
Équipements de Calcul Intensif). See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md)
for the full acknowledgment.

## License

Released under the [MIT License](LICENSE) © 2026 Mathis Delsart.

## Citation

If you use this code or any of the shipped artefacts in academic work,
please cite the project ([`CITATION.cff`](CITATION.cff) is the source of truth):

```bibtex
@software{delsart_microrts_drl_uecd_2026,
  author  = {Delsart, Mathis},
  title   = {{Deep Reinforcement Learning for Competitive Agents in MicroRTS: Architecture, Training, and Tournament Evaluation}},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.20481385},
  url     = {https://github.com/mathisdelsart/microrts-drl-uecd/releases/tag/v0.1.0},
  note    = {Master's thesis, UCLouvain},
}
```

## Author

**[Mathis Delsart](https://github.com/mathisdelsart)**, Master's thesis, UCLouvain.

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0208,50:4a0d2b,100:0d0208&height=100&section=footer" width="100%" alt="" />

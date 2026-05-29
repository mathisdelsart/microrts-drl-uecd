# Credits — Vendored Third-Party Bots

The competition bots under `microrts_agent/tournaments/competition_winners/` and the opponent
JARs under `microrts_agent/microrts/lib/bots/` are **third-party code**, included so that the
agent can be evaluated against the MicroRTS competition field reproducibly. They are **not**
covered by this project's MIT license (see [`LICENSE`](LICENSE)); copyright remains with their
original authors.

Most of these bots are published on GitHub **without an explicit license** (which, by default,
means all rights reserved). They are redistributed here in good faith for academic,
non-commercial reproducibility, **with attribution**. For any reuse beyond that, please refer to
the upstream repository and/or contact the original authors.

| Bot | Maintainer / origin | Upstream | License | Notes |
|-----|---------------------|----------|---------|-------|
| RAISocketAI | Scott Goodfriend | <https://github.com/sgoodfriend/rl-algo-impls> | **MIT** | IEEE-CoG MicroRTS winner (2023 PPO, 2024 BC+PPO) |
| CoacAI | Coac | <https://github.com/Coac/coac-ai-microrts> | none stated | Rule-based |
| Mayari | barvazkrav | <https://github.com/barvazkrav/mayariBot> | none stated | MicroRTS competition |
| MixedBot | AmoyZhp | <https://github.com/AmoyZhp/MixedBotmRTS> | none stated | IEEE-CoG 2019 (integrates Tiamat / Capivara / StrategyTactics) |
| Tiamat | Mariño, Moraes, Toledo, Lelis | <https://github.com/jr9Hernandez/TiamatBot> | none stated | 2018 mRTS competition |
| Izanagi | rubensolv | <https://github.com/rubensolv/IzanagiBot> | none stated | MicroRTS competition |
| GRojoA3N | rubensolv | <https://github.com/rubensolv/GRojoA3N> | none stated | MicroRTS competition |
| Droplet | zuozhiyang | <https://github.com/zuozhiyang/Droplet> | none stated | MicroRTS competition |
| StrategyTactics | L. H. S. Lelis et al. | microRTS competition | none stated | Won the 2017 CIG competition |
| TMA (Tactical Manager AI) | TMA authors | microRTS competition | none stated | CoG 2024 submission |
| ObiBotKenobi | — | microRTS competition | none stated | MicroRTS competition bot |
| UTS_Imass | UTS iMass team | microRTS competition | none stated | 2019 CoG MicroRTS winner |

## Engine

The base **MicroRTS** engine (`microrts_agent/microrts/microrts.jar` and the maps under
`microrts_agent/microrts/maps/`) is by **Santiago Ontañón** —
<https://github.com/santiontanon/microrts>.

> Note: "none stated" means no `LICENSE` file was found in the upstream repository at the time of
> writing. Before redistributing this repository publicly, confirm each bot's terms or obtain the
> author's permission.

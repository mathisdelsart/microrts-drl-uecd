"""Unified CLI entry point: microrts-agent <command> [args...]

Commands:
  train         Train a PPO agent
  evaluate      Evaluate an agent against a bot/agent
  tournament    Run a round-robin tournament  {run|parse|viz|analyze}
  bc            Behaviour cloning             {train|generate}
  bench         Inference benchmarks          {inference|head2head}
  analysis      Training-run analysis         {metrics|audit|params}
"""

import importlib
import sys

# command -> module exposing main(); the grouped ones delegate to their dispatcher
_COMMANDS = {
    "train": "microrts_agent.train",
    "evaluate": "microrts_agent.evaluate",
    "tournament": "microrts_agent.tournament.__main__",
    "bc": "microrts_agent.bc.__main__",
    "bench": "microrts_agent.bench.__main__",
    "analysis": "microrts_agent.analysis.__main__",
}


_HELP = """\
microrts-agent: MicroRTS Deep-RL Agent CLI (UCLouvain master's thesis).

usage: microrts-agent <command> [options]

COMMANDS

  train       Train a PPO agent from scratch or resume from a checkpoint.
              Common flags: --architecture, --total-timesteps, --map,
                            --num-bot-envs, --num-selfplay-envs, --exp-name,
                            --load-model, --resume, --seed, --with-eval
              Example:
                microrts-agent train --architecture gridnet \\
                  --total-timesteps 1000000 --num-bot-envs 8 --seed 1

  evaluate    Evaluate an agent against another agent or scripted bot.
              Plays both as P0 and P1 by default.
              Common flags: --agent, --opponent, --maps, --nb_games,
                            --max-steps, --deterministic, --record
              Example:
                microrts-agent evaluate \\
                  --agent data/agents/UECD-SingleMap-Best \\
                  --opponent CoacAI --nb_games 10 --max-steps 4000

  tournament  Round-robin tournament toolchain. Sub-commands:
                run      run a tournament from a JSON config
                parse    parse the CSV/chunks into a structured JSON
                viz      generate the PDF visualization tree
                analyze  parse + viz combined
              Example:
                microrts-agent tournament run single_map

  bc          Behaviour cloning toolchain. Sub-commands:
                generate  record demonstrations from a scripted bot
                train     train a BC agent (with optional value head)
              Example:
                microrts-agent bc train \\
                  --data 'data/BC/training/bc_chunk_*.npz' \\
                  --epochs 30 --output outputs/runs/my-bc

  bench       Inference / head-to-head decision-time benchmarks. Sub-commands:
                inference  per-tick decision time, self-play
                head2head  per-tick decision time, agent vs bot
              Example:
                microrts-agent bench inference \\
                  --agents data/agents/UECD-SingleMap-Best --games 5

  analysis    Training-run analysis utilities. Sub-commands:
                metrics  per-run PDF plots and WR summaries
                audit    sanity-check a run's files (config, log, ckpt, ...)
                params   parameter-count table per architecture / feature
              Example:
                microrts-agent analysis metrics data/agents/UECD-SingleMap-Best

For per-command flags:
  microrts-agent <command> --help

For per-sub-command flags:
  microrts-agent <command> <subcommand> --help

See README.md and examples/ for runnable demos.
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_HELP, end="")
        sys.exit(0)
    if sys.argv[1] not in _COMMANDS:
        print(f"microrts-agent: unknown command '{sys.argv[1]}'", file=sys.stderr)
        print(f"  valid commands: {' '.join(_COMMANDS)}", file=sys.stderr)
        print("  use 'microrts-agent --help' for details", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"microrts-agent {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

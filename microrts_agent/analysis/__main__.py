"""CLI dispatcher: microrts-agent analysis {metrics|audit|params} [args...]"""

import importlib
import sys

_COMMANDS = {
    "metrics": "microrts_agent.analysis.metrics",
    "audit": "microrts_agent.analysis.audit",
    "params": "microrts_agent.analysis.params",
}


_HELP = """\
microrts-agent analysis: training-run analysis utilities.

USAGE
  microrts-agent analysis <subcommand> [options]

SUBCOMMANDS

  metrics     Per-run analysis: WR curve, return, episode length, loss, KL.
              Generates a metrics/ subdir under the run with PDF plots +
              a one-line stdout summary.
              Common flags: --all, --smoothing, --max-points
              Positional: run directory path(s)
              Examples:
                microrts-agent analysis metrics data/agents/UECD-SingleMap-Best
                microrts-agent analysis metrics --all
                microrts-agent analysis metrics --smoothing 0.99 \\
                  data/ablation/feat/agent/extended_obs_s1

  audit       Sanity-check a run's files: config.json schema, train.log
              presence + step coverage, agent.pt loadability, checkpoint.pt
              presence, eval_results.csv shape. Reports missing files
              and dtype mismatches.
              Common flags: --all
              Positional: run directory path(s)
              Example:
                microrts-agent analysis audit data/agents/UECD-SingleMap-Best
                microrts-agent analysis audit --all

  params      Print parameter-count table per architecture + per feature
              (relative to the UNet-Entity-CBAM-Deep baseline). Pure
              stdout, no arguments.
              Example:
                microrts-agent analysis params

For full per-subcommand flags:
  microrts-agent analysis <subcommand> --help
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_HELP, end="")
        sys.exit(0)
    if sys.argv[1] not in _COMMANDS:
        print(f"microrts-agent analysis: unknown subcommand '{sys.argv[1]}'", file=sys.stderr)
        print(f"  valid subcommands: {' '.join(_COMMANDS)}", file=sys.stderr)
        print("  use 'microrts-agent analysis --help' for details", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"microrts-agent analysis {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

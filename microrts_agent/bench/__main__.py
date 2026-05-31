"""CLI dispatcher: microrts-agent bench {inference|head2head} [args...]"""

import importlib
import sys

_COMMANDS = {
    "inference": "microrts_agent.bench.inference",
    "head2head": "microrts_agent.bench.head_to_head",
}


_HELP = """\
microrts-agent bench: inference decision-time benchmarks.

USAGE
  microrts-agent bench <subcommand> [options]

SUBCOMMANDS

  inference   Per-tick decision-time benchmark, self-play of N agents on
              the same map. Reports min/median/max decision time per agent.
              Common flags: --agents, --games, --map, --max-steps,
                            --time-budget, --deterministic, --output
              Example:
                microrts-agent bench inference \\
                  --agents data/agents/UECD-SingleMap-Best \\
                           data/agents/GridNet-SingleMap \\
                  --games 5

  head2head   Per-tick decision-time benchmark: one RL agent vs one Java bot.
              Common flags: --agent, --opponent, --games, --map, --max-steps,
                            --time-budget, --deterministic, --output
              Example:
                microrts-agent bench head2head \\
                  --agent data/agents/UECD-SingleMap-Best \\
                  --opponent CoacAI --games 10

For full per-subcommand flags:
  microrts-agent bench <subcommand> --help
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_HELP, end="")
        sys.exit(0)
    if sys.argv[1] not in _COMMANDS:
        print(f"microrts-agent bench: unknown subcommand '{sys.argv[1]}'", file=sys.stderr)
        print(f"  valid subcommands: {' '.join(_COMMANDS)}", file=sys.stderr)
        print("  use 'microrts-agent bench --help' for details", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"microrts-agent bench {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

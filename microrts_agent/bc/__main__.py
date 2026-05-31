"""CLI dispatcher: microrts-agent bc {train|generate} [args...]"""

import importlib
import sys

_COMMANDS = {
    "train": "microrts_agent.bc.train",
    "generate": "microrts_agent.bc.generate",
}


_HELP = """\
microrts-agent bc: Behaviour cloning toolchain.

usage: microrts-agent bc <subcommand> [options]

SUBCOMMANDS

  generate    Record demonstrations from a scripted bot for BC training.
              Writes NPZ chunks of (obs, actions, rewards, dones) per game.
              Common flags: --bot, --opponents, --games-per-opponent,
                            --map, --output, --max-steps
              Example:
                microrts-agent bc generate \\
                  --bot RAISocketAI \\
                  --opponents RAISocketAI CoacAI Mayari \\
                  --games-per-opponent 100 \\
                  --map maps/open_competition/basesWorkers16x16A.xml

  train       Train a BC agent (and an optional value head) on NPZ chunks.
              Globs every bc_chunk_*.npz it is pointed at.
              Common flags: --data, --architecture, --epochs, --batch-size,
                            --lr, --vf-coef, --gamma, --seed, --gelu,
                            --output, --val-split
              Example:
                microrts-agent bc train \\
                  --data 'data/BC/training/bc_chunk_*.npz' \\
                  --architecture unet_entity_cbam_deep \\
                  --epochs 30 --batch-size 128 --lr 1e-4 \\
                  --output outputs/runs/my-bc

For full per-subcommand flags:
  microrts-agent bc <subcommand> --help
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_HELP, end="")
        sys.exit(0)
    if sys.argv[1] not in _COMMANDS:
        print(f"microrts-agent bc: unknown subcommand '{sys.argv[1]}'", file=sys.stderr)
        print(f"  valid subcommands: {' '.join(_COMMANDS)}", file=sys.stderr)
        print("  use 'microrts-agent bc --help' for details", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"microrts-agent bc {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

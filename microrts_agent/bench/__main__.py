"""CLI dispatcher: python -m microrts_agent.bench {inference|h2h} [args...]"""

import importlib
import sys

_COMMANDS = {
    "inference": "microrts_agent.bench.inference",
    "h2h": "microrts_agent.bench.h2h",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        ok = len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help")
        print(f"usage: python -m microrts_agent.bench {{{'|'.join(_COMMANDS)}}} [args...]")
        sys.exit(0 if ok else 2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"python -m microrts_agent.bench {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

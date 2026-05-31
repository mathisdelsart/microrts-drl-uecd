"""CLI dispatcher: microrts-agent bc {train|generate} [args...]"""

import importlib
import sys

_COMMANDS = {
    "train": "microrts_agent.bc.train",
    "generate": "microrts_agent.bc.generate",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        ok = len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help")
        print(f"usage: microrts-agent bc {{{'|'.join(_COMMANDS)}}} [args...]")
        sys.exit(0 if ok else 2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"microrts-agent bc {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

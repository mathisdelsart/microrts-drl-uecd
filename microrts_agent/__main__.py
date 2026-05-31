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


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        ok = len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help")
        print(f"usage: microrts-agent {{{'|'.join(_COMMANDS)}}} ...")
        sys.exit(0 if ok else 2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"microrts-agent {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

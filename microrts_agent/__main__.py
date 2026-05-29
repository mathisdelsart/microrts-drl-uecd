"""Unified CLI entry point: python -m microrts_agent <command> [args...]

Commands:
  train         Train a PPO agent
  evaluate      Evaluate an agent against a bot/agent
  tournament    Run a round-robin tournament
  bc            Behaviour cloning      {train|generate}
  bench         Inference benchmarks   {inference|head2head}
  analysis      Training-run analysis  {metrics|audit|params}
"""

import importlib
import sys

# command -> module exposing main(); the grouped ones delegate to their dispatcher
_COMMANDS = {
    "train": "microrts_agent.train",
    "evaluate": "microrts_agent.evaluate",
    "tournament": "microrts_agent.run_tournament",
    "bc": "microrts_agent.bc.__main__",
    "bench": "microrts_agent.bench.__main__",
    "analysis": "microrts_agent.analysis.__main__",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        ok = len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help")
        print(f"usage: python -m microrts_agent {{{'|'.join(_COMMANDS)}}} ...")
        sys.exit(0 if ok else 2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"python -m microrts_agent {cmd}"
    importlib.import_module(_COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()

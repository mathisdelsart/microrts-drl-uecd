"""
External server management for tournament bots (UTS_Imass).
"""

import os
import subprocess
import time
from pathlib import Path

from microrts_agent.paths import AGENT_DIR, PROJECT_ROOT

UTS_IMASS_PORT = 9823
UTS_IMASS_SCRIPT = AGENT_DIR / "bots" / "UTS_Imass" / "start_server.sh"
UTS_IMASS_TRAINING_DIR = AGENT_DIR / "bots" / "UTS_Imass" / "training_data"


def _open_server_log(path):
    """Open the server log file; the returned handle is owned by the Popen wrapper."""
    return open(path, "a")


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is accepting connections."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _kill_existing_server(port: int):
    """Kill any existing process LISTENING on the given port.

    Uses -sTCP:LISTEN to only target the server process, not client
    connections (e.g. our own JVM connected as a client).
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", "-sTCP:LISTEN", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = result.stdout.strip().split()
        my_pid = os.getpid()
        for pid in pids:
            if not pid:
                continue
            pid_int = int(pid)
            if pid_int == my_pid:
                continue  # never kill ourselves
            os.kill(pid_int, 9)
        if pids:
            time.sleep(1)
            print(f"  Killed existing server on port {port} (PID {', '.join(pids)})")
    except Exception:
        pass


def _is_slurm_array() -> bool:
    """Detect if running inside a SLURM array job."""
    return "SLURM_ARRAY_TASK_ID" in os.environ


def needs_uts_imass(config_or_names) -> bool:
    """Check if any AI is UTS_Imass.

    Accepts a TournamentConfig (uses .ai_names) or a list of bot name strings.
    """
    names = config_or_names.ai_names if hasattr(config_or_names, "ai_names") else config_or_names
    return any(name.lower().replace("_", "") == "utsimass" for name in names)


def start_uts_imass_server() -> subprocess.Popen | None:
    """Start UTS_Imass server.

    Local mode: kills any stale server first, starts fresh.
    Cluster mode (SLURM array): reuses existing server if running.
    """
    if _is_port_open(UTS_IMASS_PORT):
        if _is_slurm_array():
            print(f"  UTS_Imass server already running on port {UTS_IMASS_PORT} (shared)")
            return None
        else:
            _kill_existing_server(UTS_IMASS_PORT)

    if not UTS_IMASS_SCRIPT.exists():
        print(f"  WARNING: UTS_Imass start_server.sh not found at {UTS_IMASS_SCRIPT}")
        return None

    os.makedirs(UTS_IMASS_TRAINING_DIR, exist_ok=True)

    print(f"  Starting UTS_Imass server on port {UTS_IMASS_PORT}...")
    print(f"  Training dir: {UTS_IMASS_TRAINING_DIR}/{{map_name}}/")
    log_path = str(UTS_IMASS_TRAINING_DIR / "server.log")
    log_file = _open_server_log(log_path)
    proc = subprocess.Popen(
        ["bash", str(UTS_IMASS_SCRIPT), "--timeout", "7200"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    proc._log_file = log_file

    for i in range(30):
        time.sleep(1)
        if _is_port_open(UTS_IMASS_PORT):
            print(f"  UTS_Imass server ready (took {i + 1}s)")
            return proc

    print("  WARNING: UTS_Imass server did not start within 30s")
    proc.terminate()
    log_file.close()
    return None


def stop_uts_imass_server(proc: subprocess.Popen | None):
    """Stop the UTS_Imass server.

    Only stops if we started it (proc is not None).
    In SLURM array mode, never kills -- other chunks may still need it.
    """
    if proc is None:
        return
    if _is_slurm_array():
        print("  UTS_Imass server left running (SLURM array mode).")
        if hasattr(proc, "_log_file"):
            proc._log_file.close()
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    if hasattr(proc, "_log_file"):
        proc._log_file.close()
    if _is_port_open(UTS_IMASS_PORT):
        _kill_existing_server(UTS_IMASS_PORT)
    print("  UTS_Imass server stopped.")


def run_uts_imass_preanalysis(ai, map_path: str, gs, budget_ms: int = 3600000):
    """Run preGameAnalysis for a UTS_Imass AI instance (same logic as env_pool.run_preanalysis).

    Each new Java AI instance opens a fresh socket → new server thread with no
    loaded data. Must call preGameAnalysis to register the training_data folder
    so the server loads cached opening strategies.
    """
    from jpype.types import JLong

    map_stem = Path(map_path).stem
    map_dir = UTS_IMASS_TRAINING_DIR / map_stem
    cached = map_dir.is_dir() and any(f.name.endswith("_config.json") for f in map_dir.iterdir())
    os.makedirs(map_dir, exist_ok=True)
    rel_folder = str(map_dir.resolve().relative_to(Path.cwd()))

    UTS_MIN_BUDGET_MS = 300_000
    if cached:
        effective_budget = 1000
        label = f"cached ({map_dir.name}/)"
    elif budget_ms < UTS_MIN_BUDGET_MS:
        print(
            f"    Pre-analysis: UtsImass on {map_stem} "
            f"-- skipped (budget {budget_ms / 1000:.0f}s < 5min minimum)"
        )
        return
    else:
        effective_budget = int(budget_ms)
        label = f"{budget_ms / 1000:.0f}s budget"

    print(f"    Pre-analysis: UtsImass on {map_stem} -- {label}")
    t0 = time.time()
    ai.preGameAnalysis(gs, JLong(effective_budget), rel_folder)
    print(f"    Pre-analysis done ({time.time() - t0:.1f}s)")

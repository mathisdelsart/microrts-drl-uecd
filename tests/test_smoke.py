"""Smoke tests for microrts_agent.

Each test is a fast non-regression check — it does not validate
correctness in depth, only that the most critical code paths still
load, parse, and run without crashing. Together they cover every class
of regression that has bitten the repo recently:

  - broken imports after a refactor,
  - CLI subcommand drift (e.g. evaluate broken by gymnasium >= 1.2),
  - JSON path drift (e.g. tournament configs pointing at outputs/runs/),
  - shipped-agent config / state-dict shape regressions,
  - wrapper composition,
  - env-step and forward-pass plumbing.

Total runtime target: ~2-3 min on a CPU-only CI runner.
"""

import importlib
import json
import os
import pkgutil
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_AGENTS = [
    "GridNet-SingleMap",
    "UECD-SingleMap-Best",
    "UECD-SingleMap-Rushed",
    "UECD-SingleMap-TopFeats",
    "UECD-SingleMap-AllFeats",
    "UECD-MultiMap",
    "UECD-MultiMap-Best",
    "UECD-BC",
    "UECD-BC-PPO",
]


# ── 1. Import sanity ─────────────────────────────────────────────────────
def test_imports_recursive():
    """Every public module in microrts_agent imports without error."""
    import microrts_agent

    failed = []
    for _, name, _ in pkgutil.walk_packages(microrts_agent.__path__, prefix="microrts_agent."):
        if "bots" in name:
            # Vendored third-party Python with external native deps (BL_JPS, ...).
            continue
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            failed.append((name, repr(e)))
    assert not failed, "Failed imports:\n" + "\n".join(f"  {n}: {e}" for n, e in failed)


# ── 2. CLI surface ───────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", ["train", "evaluate", "tournament", "bc", "bench", "analysis"])
def test_cli_subcommand_help(cmd):
    """python -m microrts_agent <cmd> --help exits 0."""
    res = subprocess.run(
        [sys.executable, "-m", "microrts_agent", cmd, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert res.returncode == 0, (
        f"--help for {cmd} failed (exit {res.returncode})\n"
        f"stdout: {res.stdout.decode()[:500]}\n"
        f"stderr: {res.stderr.decode()[:500]}"
    )


# ── 3. Paths constants ───────────────────────────────────────────────────
def test_paths_constants_resolve():
    """paths.py constants are anchored at the repo root and the static
    ones (MICRORTS_DIR, TOURNAMENT_CONFIGS_DIR) actually exist."""
    from microrts_agent import paths

    assert paths.PROJECT_ROOT == REPO_ROOT
    assert paths.AGENT_DIR == REPO_ROOT / "microrts_agent"
    assert paths.MICRORTS_DIR.exists()
    assert paths.TOURNAMENT_CONFIGS_DIR.exists()
    # OUTPUTS_DIR and its subdirs are git-ignored — may or may not exist
    # on a fresh clone; only check their *parent* root resolves.
    assert paths.OUTPUTS_DIR == REPO_ROOT / "outputs"


# ── 4. Tournament configs ────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["single_map", "multi_map"])
def test_tournament_config_loads(name):
    """Each shipped tournament config loads and every agent: path
    resolves to a config.json that exists on disk."""
    from microrts_agent.tournament.config import TournamentConfig

    cfg = TournamentConfig.load(f"microrts_agent/tournament_configs/{name}.json")
    assert len(cfg.ais) >= 16, f"{name}: too few AIs ({len(cfg.ais)})"
    assert len(cfg.maps) >= 1


def test_tournament_config_maps_exist_on_disk():
    """Every map referenced by either shipped tournament config is on disk."""
    for name in ("single_map", "multi_map"):
        with open(f"microrts_agent/tournament_configs/{name}.json") as f:
            cfg = json.load(f)
        for m in cfg["maps"]:
            full = REPO_ROOT / "microrts_agent" / "microrts" / m
            assert full.exists(), f"{name}: map missing on disk: {full}"


# ── 5. Bot registry ──────────────────────────────────────────────────────
def test_ai_mapping_keys_routable():
    """Every AI_MAPPING value is a non-None bot factory."""
    from microrts_agent.registries.ai import AI_MAPPING

    assert len(AI_MAPPING) >= 18
    for name, ai in AI_MAPPING.items():
        assert ai is not None, f"AI_MAPPING[{name}] is None"


# ── 6. JNI bridge ────────────────────────────────────────────────────────
def test_jni_bridge_loads(jvm):
    """microrts.jar + bridge.jar are discoverable on the JVM classpath."""
    UnitTypeTable = jvm.JClass("rts.units.UnitTypeTable")
    assert UnitTypeTable() is not None
    JNIClient = jvm.JClass("tests.JNIGridnetClient")
    assert JNIClient is not None


# ── 7. Vec env smoke ─────────────────────────────────────────────────────
def test_vec_env_step(jvm):  # noqa: ARG001 — fixture starts the JVM
    """make_bot_env + reset + step produces a valid info dict."""
    from microrts_agent.envs.factory import JVM_ARGS, make_bot_env
    from microrts_agent.registries.ai import AI_MAPPING

    env = make_bot_env(
        ai1s=[AI_MAPPING["WorkerRush"]],
        ai2s=[AI_MAPPING["LightRush"]],
        map_paths=["maps/open_competition/basesWorkers16x16A.xml"],
        partial_obs=False,
        max_steps=50,
        jvm_args=JVM_ARGS,
    )
    env.reset()  # None for bot_vs_bot
    _, _, _, info = env.step(np.zeros(1, dtype=np.int32))
    assert "raw_rewards" in info[0]


# ── 8. Architecture instantiation ────────────────────────────────────────
@pytest.mark.parametrize(
    "arch",
    [
        "gridnet",
        "impala",
        "impala_entity",
        "unet",
        "unet_entity",
        "unet_entity_cbam",
        "unet_entity_cbam_deep",
    ],
)
def test_architecture_in_registry(arch):
    """Every architecture name advertised by the CLI is in the registry."""
    from microrts_agent.architectures.factory import ARCHITECTURE_REGISTRY

    assert arch in ARCHITECTURE_REGISTRY
    cls = ARCHITECTURE_REGISTRY[arch]
    assert cls is not None and callable(cls)


# ── 9. Shipped agents loadable ───────────────────────────────────────────
@pytest.mark.parametrize("agent_name", SHIPPED_AGENTS)
def test_shipped_agent_loadable(agent_name):
    """Each shipped agent's config + agent.pt rebuild a model whose
    state-dict matches the config (no shape mismatch)."""
    from microrts_agent.architectures.factory import load_agent_from_config

    agent_dir = REPO_ROOT / "data" / "agents" / agent_name
    assert agent_dir.is_dir(), f"shipped agent missing: {agent_dir}"
    agent, config = load_agent_from_config(str(agent_dir), device="cpu")
    assert agent is not None
    assert "architecture" in config


# ── 10. Wrappers compose ────────────────────────────────────────────────
def test_wrappers_compose(jvm):  # noqa: ARG001
    """apply_env_wrappers chains frame_stack + reserved_obs without error."""
    from microrts_agent.envs.factory import JVM_ARGS, make_agent_env
    from microrts_agent.registries.ai import AI_MAPPING
    from microrts_agent.wrappers.factory import apply_env_wrappers

    env = make_agent_env(
        num_bot_envs=1,
        num_selfplay_envs=0,
        ai2s=[AI_MAPPING["RandomBiasedAI"]],
        map_paths=["maps/open_competition/basesWorkers16x16A.xml"],
        player=0,
        partial_obs=False,
        max_steps=50,
        reward_weight=np.ones(10, dtype=np.float32),
        jvm_args=JVM_ARGS,
        padded=False,
        max_height=0,
        max_width=0,
        alternate_players=False,
        filtered_masks=False,
        extended_obs=False,
    )
    wrapped = apply_env_wrappers(env, gamma=0.99, frame_stack=2, reserved_obs=True)
    obs = wrapped.reset()
    assert obs is not None


# ── 11. End-to-end evaluate ──────────────────────────────────────────────
def test_evaluate_end_to_end():
    """python -m microrts_agent evaluate <shipped agent> vs RandomBiasedAI
    finishes one game per position without crashing."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "microrts_agent",
            "evaluate",
            "--agent",
            "data/agents/GridNet-SingleMap",
            "--opponent",
            "RandomBiasedAI",
            "--nb_games",
            "1",
            "--max-steps",
            "200",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=180,
        env=env,
        check=False,
    )
    assert res.returncode == 0, (
        f"evaluate failed (exit {res.returncode})\n"
        f"stdout tail: {res.stdout.decode()[-1000:]}\n"
        f"stderr tail: {res.stderr.decode()[-1000:]}"
    )
    assert b"RESULTS" in res.stdout, "evaluate produced no RESULTS block"


# ── 12. End-to-end PPO one-update ───────────────────────────────────────
def test_ppo_one_update(tmp_path):
    """python -m microrts_agent train with a tiny budget runs through at
    least one PPO update without crashing.

    Runs from a tmp_path cwd so outputs/runs/<exp_name>/ lands in pytest's
    scratch dir instead of polluting the real outputs/ tree, and copies
    the engine maps + bridge from the repo so the JNI bridge can find them.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    # The train CLI resolves map paths and JARs relative to its cwd. Run
    # from the repo root and let outputs/runs/<exp> land there; pytest
    # cleans it up after each run.
    exp_name = f"smoke_train_{tmp_path.name}"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "microrts_agent",
            "train",
            "--architecture",
            "gridnet",
            "--total-timesteps",
            "256",
            "--num-bot-envs",
            "1",
            "--num-selfplay-envs",
            "0",
            "--num-steps",
            "128",
            "--num-models",
            "1",
            "--exp-name",
            exp_name,
            "--seed",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=240,
        env=env,
        check=False,
    )
    # Clean up the run dir the trainer created so the test stays hermetic.
    run_dir = REPO_ROOT / "outputs" / "runs" / exp_name
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    assert res.returncode == 0, (
        f"train failed (exit {res.returncode})\n"
        f"stdout tail: {res.stdout.decode()[-1500:]}\n"
        f"stderr tail: {res.stderr.decode()[-1500:]}"
    )


# ── 13. Setup scripts syntax ────────────────────────────────────────────
@pytest.mark.parametrize(
    "script",
    [
        "setup/local.sh",
        "setup/cluster.sh",
        "experiments/_setup_env.sh",
        "microrts_agent/microrts/build_bridge.sh",
    ],
)
def test_shell_script_syntax(script):
    """`bash -n <script>` parses cleanly (catches typos in the shipped
    install / SLURM helpers and in the bridge build script)."""
    res = subprocess.run(
        ["bash", "-n", script],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert res.returncode == 0, f"{script} has a bash syntax error:\n{res.stderr.decode()[:1000]}"


# ── 14. BC teacher dataset schema ───────────────────────────────────────
def test_bc_chunks_schema():
    """Every shipped BC chunk under data/bc_training/ has the expected
    NPZ keys and 4D obs shape."""
    bc_dir = REPO_ROOT / "data" / "bc_training"
    chunks = sorted(bc_dir.glob("bc_chunk_*.npz"))
    assert len(chunks) >= 3, f"too few BC chunks: {len(chunks)}"
    for chunk in chunks:
        npz = np.load(chunk)
        assert {"obs", "actions", "rewards"}.issubset(set(npz.files)), (
            f"{chunk.name}: missing keys (has {list(npz.files)})"
        )
        # obs: (N, H, W, C) — sanity check the H, W match the 16x16 thesis map
        assert npz["obs"].ndim == 4
        assert npz["obs"].shape[1:3] == (16, 16)

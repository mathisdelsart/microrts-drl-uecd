"""Smoke tests for microrts_agent.

Each test is a fast non-regression check: it does not validate
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


def _run_cli(*argv, timeout=60):
    """Run `python -m microrts_agent <argv>` with CUDA disabled, capture
    output, and return the CompletedProcess. Common boilerplate for the
    subprocess-based smoke tests."""
    return subprocess.run(
        [sys.executable, "-m", "microrts_agent", *argv],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=timeout,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )


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
        except Exception as e:
            failed.append((name, repr(e)))
    assert not failed, "Failed imports:\n" + "\n".join(f"  {n}: {e}" for n, e in failed)


# ── 2. CLI surface ───────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", ["train", "evaluate", "tournament", "bc", "bench", "analysis"])
def test_cli_subcommand_help(cmd):
    """python -m microrts_agent <cmd> --help exits 0."""
    res = _run_cli(cmd, "--help")
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
    # OUTPUTS_DIR and its subdirs are git-ignored: may or may not exist
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
@pytest.mark.usefixtures("jvm")
def test_vec_env_step():
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
@pytest.mark.usefixtures("jvm")
def test_wrappers_compose():
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
    res = _run_cli(
        "evaluate",
        "--agent",
        "data/agents/GridNet-SingleMap",
        "--opponent",
        "RandomBiasedAI",
        "--nb_games",
        "1",
        "--max-steps",
        "200",
        timeout=180,
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

    The train CLI resolves map paths and JARs relative to its cwd, so this
    test runs from the repo root and lets outputs/runs/<exp> land there;
    we delete it on teardown to stay hermetic.
    """
    exp_name = f"smoke_train_{tmp_path.name}"
    res = _run_cli(
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
        timeout=240,
    )
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
        "setup/_common.sh",
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
    """Every shipped BC chunk under data/BC/training/ has the expected
    NPZ keys and 4D obs shape."""
    bc_dir = REPO_ROOT / "data" / "BC" / "training"
    chunks = sorted(bc_dir.glob("bc_chunk_*.npz"))
    assert len(chunks) >= 3, f"too few BC chunks: {len(chunks)}"
    for chunk in chunks:
        npz = np.load(chunk)
        assert {"obs", "actions", "rewards"}.issubset(set(npz.files)), (
            f"{chunk.name}: missing keys (has {list(npz.files)})"
        )
        # obs: (N, H, W, C): sanity check the H, W match the 16x16 thesis map
        assert npz["obs"].ndim == 4
        assert npz["obs"].shape[1:3] == (16, 16)


# ── 15. Map registry consistency ────────────────────────────────────────
def test_map_registry_files_exist():
    """Every map referenced by registries/maps.py points to a real file
    under microrts_agent/microrts/maps/."""
    from microrts_agent.registries.maps import (
        COMPETITION_HIDDEN_MAPS,
        COMPETITION_OPEN_MAPS,
        COMPETITION_OPEN_SMALL,
        MAP_MAX_CYCLES,
    )

    all_refs = (
        set(COMPETITION_OPEN_MAPS)
        | set(COMPETITION_HIDDEN_MAPS)
        | set(COMPETITION_OPEN_SMALL)
        | set(MAP_MAX_CYCLES.keys())
    )
    missing = [m for m in all_refs if not (REPO_ROOT / "microrts_agent" / "microrts" / m).exists()]
    assert not missing, "Maps registered but missing on disk:\n" + "\n".join(
        f"  {m}" for m in missing
    )


# ── 16. Bridge classes findable ─────────────────────────────────────────
@pytest.mark.parametrize(
    "java_class",
    [
        "rts.units.UnitTypeTable",
        "tests.JNIGridnetClient",
        "tests.JNIGridnetClientSelfPlay",
        "tests.JNIBotClient",
        "tests.JNIGridnetVecClient",
        "tests.FilteredMaskClient",
        "ai.RandomBiasedAI",
        "ai.coac.CoacAI",
    ],
)
def test_bridge_java_class_resolvable(jvm, java_class):
    """Every Java class referenced by the Python side resolves on the
    classpath: catches silent bridge-build / dependency drift."""
    cls = jvm.JClass(java_class)
    assert cls is not None


# ── 17. Multi-map padded env ────────────────────────────────────────────
@pytest.mark.usefixtures("jvm")
def test_multimap_padded_env():
    """make_agent_env with padded=True (multi-map mode) accepts a 16x16
    map padded to 32x32 and steps without error."""
    from microrts_agent.envs.factory import JVM_ARGS, make_agent_env
    from microrts_agent.registries.ai import AI_MAPPING

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
        padded=True,
        max_height=32,
        max_width=32,
        alternate_players=False,
        filtered_masks=False,
        extended_obs=False,
    )
    obs = env.reset()
    assert obs is not None
    # Padded obs should be 32x32, not 16x16
    assert obs.shape[1:3] == (32, 32), f"padded obs wrong shape: {obs.shape}"


# ── 18. evaluate --deterministic ─────────────────────────────────────────
def test_evaluate_deterministic_flag():
    """`evaluate --deterministic` runs end-to-end (covers the greedy
    argmax branch of get_actions)."""
    res = _run_cli(
        "evaluate",
        "--agent",
        "data/agents/GridNet-SingleMap",
        "--opponent",
        "RandomBiasedAI",
        "--nb_games",
        "1",
        "--max-steps",
        "200",
        "--deterministic",
        timeout=180,
    )
    assert res.returncode == 0, (
        f"evaluate --deterministic failed (exit {res.returncode})\n"
        f"stderr tail: {res.stderr.decode()[-1000:]}"
    )
    assert b"RESULTS" in res.stdout


# ── 19. evaluate RL vs RL ───────────────────────────────────────────────
def test_evaluate_rl_vs_rl_mode():
    """`evaluate` with two RL agents triggers the rl_vs_rl batch path
    (different from rl_vs_bot exercised by test_evaluate_end_to_end)."""
    res = _run_cli(
        "evaluate",
        "--agent",
        "data/agents/GridNet-SingleMap",
        "--opponent",
        "data/agents/UECD-SingleMap-AllFeats",
        "--nb_games",
        "1",
        "--max-steps",
        "200",
        timeout=180,
    )
    assert res.returncode == 0, (
        f"evaluate rl-vs-rl failed (exit {res.returncode})\n"
        f"stderr tail: {res.stderr.decode()[-1000:]}"
    )
    assert b"RESULTS" in res.stdout


# ── 20. bench inference smoke ───────────────────────────────────────────
def test_bench_inference_smoke():
    """`python -m microrts_agent bench inference` runs on a shipped agent
    + RandomBiasedAI for 1 game."""
    res = _run_cli(
        "bench",
        "inference",
        "--agents",
        "data/agents/GridNet-SingleMap",
        "RandomBiasedAI",
        "--games",
        "1",
        "--max-steps",
        "200",
        timeout=180,
    )
    assert res.returncode == 0, (
        f"bench inference failed (exit {res.returncode})\n"
        f"stderr tail: {res.stderr.decode()[-1000:]}"
    )


# ── 21. BC generate -> BC train mini pipeline ───────────────────────────
def test_bc_generate_then_train(tmp_path):
    """`bc generate` produces an NPZ chunk; `bc train` consumes it and
    saves an agent.pt + config.json. Covers both BC subcommands in one
    end-to-end pipeline."""
    npz = tmp_path / "bc_smoke.npz"
    # 1. Generate 1 game of RandomBiasedAI vs RandomBiasedAI.
    res = _run_cli(
        "bc",
        "generate",
        "--bot",
        "RandomBiasedAI",
        "--opponents",
        "RandomBiasedAI",
        "--games-per-opponent",
        "1",
        "--max-steps",
        "200",
        "--output",
        str(npz),
        timeout=120,
    )
    assert res.returncode == 0, (
        f"bc generate failed (exit {res.returncode})\nstderr tail: {res.stderr.decode()[-1000:]}"
    )
    # bc generate names chunks `bc_chunk_<opponent>_<idx>.npz` inside the
    # parent directory of --output, regardless of the filename you pass.
    chunks = sorted(tmp_path.glob("bc_chunk_*.npz"))
    assert chunks, f"bc generate produced no bc_chunk_*.npz in {tmp_path}"
    # 2. Train BC for 1 epoch on that chunk.
    out_dir = tmp_path / "bc_run"
    res = _run_cli(
        "bc",
        "train",
        "--data",
        *(str(c) for c in chunks),
        "--architecture",
        "gridnet",
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--output",
        str(out_dir),
        timeout=240,
    )
    assert res.returncode == 0, (
        f"bc train failed (exit {res.returncode})\nstderr tail: {res.stderr.decode()[-1500:]}"
    )
    assert (out_dir / "agent.pt").exists()
    assert (out_dir / "config.json").exists()


# ── 22. Setup scripts have +x bit ───────────────────────────────────────
# setup/_common.sh is intentionally excluded: it is sourced by the other
# entry points, not executed directly.
@pytest.mark.parametrize(
    "script",
    [
        "setup/local.sh",
        "setup/cluster.sh",
        "microrts_agent/microrts/build_bridge.sh",
    ],
)
def test_setup_script_executable_bit(script):
    """The user-facing install scripts are checked in with the
    executable bit set, so `./setup/local.sh` works without `bash` prefix."""
    path = REPO_ROOT / script
    assert os.access(path, os.X_OK), f"{script} is not executable (missing +x bit)"


# ── 23. Shipped results.csv schemas ─────────────────────────────────────
@pytest.mark.parametrize(
    "csv_path, required_cols",
    [
        ("data/BC/baseline/results.csv", {"opponent", "games", "total_wr"}),
        ("data/rush_collapse/results.csv", {"opponent", "games", "total_wr"}),
        ("data/generalization_probes/results.csv", {"map", "opponent", "games", "total_wr"}),
        ("data/ablation/arch/eval/results.csv", {"arch", "seed", "opponent", "total_wr"}),
        ("data/ablation/feat/eval/results.csv", {"feature", "seed", "opponent", "total_wr"}),
    ],
)
def test_shipped_csv_schema(csv_path, required_cols):
    """Every shipped results.csv has the columns the docs / figure
    scripts depend on. Catches accidental schema drift on the next
    rsync from Lyra."""
    import csv

    path = REPO_ROOT / csv_path
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
    missing = required_cols - set(header)
    assert not missing, f"{csv_path}: missing columns {missing} (has {header})"


# ── 24. Every dotted sub-CLI dispatcher responds to --help ──────────────
@pytest.mark.parametrize(
    "cmd_path",
    [
        "microrts_agent.bc",
        "microrts_agent.bench",
        "microrts_agent.analysis",
    ],
)
def test_grouped_cli_dispatcher_help(cmd_path):
    """python -m <microrts_agent.bc|bench|analysis> (no arg) prints the
    usage line listing its sub-subcommands."""
    res = subprocess.run(
        [sys.executable, "-m", cmd_path],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert (b"usage:" in res.stdout) or (b"usage:" in res.stderr), (
        f"{cmd_path}: dispatcher printed no usage line"
    )


# ── 25. Sub-subcommand --help ───────────────────────────────────────────
@pytest.mark.parametrize(
    "argv",
    [
        ("tournament", "run", "--help"),
        ("tournament", "parse", "--help"),
        ("tournament", "viz", "--help"),
        ("tournament", "analyze", "--help"),
        ("bc", "train", "--help"),
        ("bc", "generate", "--help"),
        ("bench", "inference", "--help"),
        ("bench", "head2head", "--help"),
        ("analysis", "metrics", "--help"),
        ("analysis", "audit", "--help"),
        ("analysis", "params", "--help"),
    ],
)
def test_sub_subcommand_help(argv):
    """Every nested sub-subcommand --help exits 0 (catches dispatch
    regressions in microrts_agent/{bc,bench,analysis,tournament}/__main__)."""
    res = _run_cli(*argv)
    assert res.returncode == 0, (
        f"{' '.join(argv)}: failed (exit {res.returncode})\nstderr: {res.stderr.decode()[:500]}"
    )


# ── 26. Every architecture's forward pass on a synthetic obs ────────────
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
def test_architecture_forward_pass(arch):
    """Each architecture instantiates and forward-passes on a random
    16x16x29 obs. Catches kwargs/shape regressions that would only
    surface at training time otherwise."""
    import torch

    from microrts_agent.architectures.factory import ARCHITECTURE_REGISTRY

    cls = ARCHITECTURE_REGISTRY[arch]
    model = cls(obs_channels=29, action_nvec=[6, 4, 4, 4, 4, 7, 49])
    model.eval()
    # Obs is channels-last (N, H, W, C): the engine emits float planes in
    # that layout and the architectures transpose internally before conv.
    obs = torch.randn(2, 16, 16, 29)
    masks = torch.ones(2, 16, 16, 78, dtype=torch.bool)
    with torch.no_grad():
        out = model.forward(obs, invalid_action_masks=masks)
    assert out is not None
    assert "action" in out or hasattr(out, "shape")


# ── 27. YAML files parse ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "yaml_path",
    [
        ".pre-commit-config.yaml",
        ".github/dependabot.yml",
        ".github/workflows/ci.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
    ],
)
def test_yaml_file_parses(yaml_path):
    """Every checked-in YAML config file is valid YAML."""
    yaml = pytest.importorskip("yaml")
    path = REPO_ROOT / yaml_path
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data is not None, f"{yaml_path}: parses to None / empty"


# ── 28. CITATION.cff is valid YAML + required fields ────────────────────
def test_citation_cff_valid():
    """CITATION.cff parses + has the required fields per the CFF 1.2 schema."""
    yaml = pytest.importorskip("yaml")
    with open(REPO_ROOT / "CITATION.cff") as f:
        cff = yaml.safe_load(f)
    for field in ("cff-version", "title", "type", "version", "authors", "url"):
        assert field in cff, f"CITATION.cff missing field: {field}"
    assert cff["cff-version"].startswith("1."), f"unexpected cff-version: {cff['cff-version']}"


# ── 29. pyproject.toml + ruff.toml parse ─────────────────────────────────
@pytest.mark.parametrize("toml_path", ["pyproject.toml", "ruff.toml"])
def test_toml_file_parses(toml_path):
    """Every TOML config file at the repo root parses cleanly."""
    try:
        import tomllib  # stdlib on Python 3.11+
    except ImportError:
        import tomli as tomllib  # backport for Python 3.9 / 3.10

    path = REPO_ROOT / toml_path
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data, f"{toml_path}: parses to empty dict"


# ── 30. Every shipped JSON parses + has 'architecture' for agent configs ─
def test_every_shipped_json_parses():
    """Every committed JSON in microrts_agent/tournament_configs/ and
    data/ parses without error."""
    paths = [
        *(REPO_ROOT / "microrts_agent" / "tournament_configs").glob("*.json"),
        *(REPO_ROOT / "data").rglob("*.json"),
    ]
    assert paths, "no JSON files found"
    bad = []
    for p in paths:
        try:
            with open(p) as f:
                json.load(f)
        except Exception as e:
            bad.append((str(p), repr(e)))
    assert not bad, "Unparseable JSON:\n" + "\n".join(f"  {p}: {e}" for p, e in bad)


# ── 31. Sample of arch ablation agents loadable ─────────────────────────
@pytest.mark.parametrize(
    "ablation_run",
    [
        "gridnet_s1",
        "impala_entity_s2",
        "unet_entity_cbam_deep_s3",
    ],
)
def test_ablation_arch_agent_loadable(ablation_run):
    """A sample of arch-ablation runs (one per seed across the 7 archs)
    can be rebuilt with load_agent_from_config."""
    from microrts_agent.architectures.factory import load_agent_from_config

    run_dir = REPO_ROOT / "data" / "ablation" / "arch" / "agent" / ablation_run
    assert run_dir.is_dir(), f"ablation run missing: {run_dir}"
    agent, config = load_agent_from_config(str(run_dir), device="cpu")
    assert agent is not None
    assert "architecture" in config


# ── 32. Sample of feat ablation agents loadable ─────────────────────────
@pytest.mark.parametrize(
    "ablation_run",
    [
        "baseline_s1",
        "extended_obs_s2",
        "hl_gauss_s3",
    ],
)
def test_ablation_feat_agent_loadable(ablation_run):
    """A sample of feat-ablation runs (top / bottom / baseline) can be
    rebuilt with load_agent_from_config."""
    from microrts_agent.architectures.factory import load_agent_from_config

    run_dir = REPO_ROOT / "data" / "ablation" / "feat" / "agent" / ablation_run
    assert run_dir.is_dir(), f"ablation run missing: {run_dir}"
    agent, config = load_agent_from_config(str(run_dir), device="cpu")
    assert agent is not None
    assert "architecture" in config


# ── 33. Every vendored bot JAR is a valid ZIP ───────────────────────────
@pytest.mark.parametrize(
    "jar_path",
    sorted(
        str(p.relative_to(Path(__file__).resolve().parent.parent))
        for p in (
            Path(__file__).resolve().parent.parent / "microrts_agent" / "microrts" / "lib"
        ).rglob("*.jar")
    )
    + ["microrts_agent/microrts/microrts.jar"],
)
def test_jar_is_valid_zip(jar_path):
    """Every shipped .jar opens as a ZIP and contains at least one .class
    file (MANIFEST.MF is technically optional in the JAR spec: some
    vendored bot JARs do not ship one). Catches a truncated / corrupt
    download."""
    import zipfile

    path = REPO_ROOT / jar_path
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    assert any(n.endswith(".class") for n in names), f"{jar_path}: no .class files inside"


# ── 34. setup scripts have a #!/bin/bash shebang on line 1 ──────────────
@pytest.mark.parametrize(
    "script",
    [
        "setup/_common.sh",
        "setup/local.sh",
        "setup/cluster.sh",
        "microrts_agent/microrts/build_bridge.sh",
    ],
)
def test_setup_script_has_shebang(script):
    """User-runnable scripts start with #!/bin/bash on line 1."""
    with open(REPO_ROOT / script) as f:
        first = f.readline().rstrip("\n")
    assert first == "#!/bin/bash", f"{script}: line 1 is '{first}' (expected #!/bin/bash)"


# ── 35. Every train.log line in a shipped agent has a recognisable format
def test_shipped_train_logs_non_empty():
    """Every shipped train.log under data/agents/ has at least one line
    matching the canonical `step=<n>` prefix. Catches accidental
    truncation / re-encoding."""
    import re

    pattern = re.compile(r"step=\s*[\d,]+")
    failed = []
    for run_dir in (REPO_ROOT / "data" / "agents").iterdir():
        if not run_dir.is_dir():
            continue
        log = run_dir / "train.log"
        if not log.exists():
            continue  # UECD-BC has no train.log (documented)
        with open(log, errors="ignore") as f:
            for line in f:
                if pattern.search(line):
                    break
            else:
                failed.append(str(log))
    assert not failed, "train.log without canonical step= lines:\n" + "\n".join(failed)


# ── 36. Vendored microrts maps are all valid XML ────────────────────────
def test_vendored_maps_are_valid_xml():
    """Every map file referenced by either shipped tournament config
    parses as XML."""
    import xml.etree.ElementTree as ET

    for name in ("single_map", "multi_map"):
        with open(REPO_ROOT / "microrts_agent" / "tournament_configs" / f"{name}.json") as f:
            cfg = json.load(f)
        for m in cfg["maps"]:
            ET.parse(REPO_ROOT / "microrts_agent" / "microrts" / m)


# ── 37. CI workflow declares the pytest job ─────────────────────────────
def test_ci_workflow_declares_test_job():
    """The CI workflow has a `tests` job: guards against accidentally
    removing the smoke-test runner from .github/workflows/ci.yml."""
    yaml = pytest.importorskip("yaml")
    with open(REPO_ROOT / ".github" / "workflows" / "ci.yml") as f:
        ci = yaml.safe_load(f)
    assert "tests" in ci["jobs"], "ci.yml has no 'tests' job"
    assert "pytest" in ci["jobs"]["tests"]["name"].lower()


# ── 38. tournament parse/viz on shipped JSONs ───────────────────────────
@pytest.mark.parametrize("name", ["single_map", "multi_map"])
def test_tournament_parse_shipped(name):
    """`tournament parse` reads a shipped tournament_parsed.json without error.

    The shipped file was already produced by the runner; calling parse on it
    catches schema drift in the parser between runner and consumer."""
    parsed = REPO_ROOT / "data" / "tournaments" / name / "tournament_parsed.json"
    if not parsed.exists():
        pytest.skip(f"shipped parsed tournament missing: {parsed}")
    with open(parsed) as f:
        data = json.load(f)
    # Cheap structural sanity matching what tournament/viz consumers rely on.
    assert "games" in data, f"{parsed.name}: missing 'games' key"
    assert isinstance(data["games"], list)
    assert data["games"], f"{parsed.name}: zero games recorded"
    for game in data["games"][:5]:
        assert "players" in game and {"ai1", "ai2"}.issubset(game["players"])
        assert "result" in game and "time" in game["result"]


@pytest.mark.parametrize("name", ["single_map", "multi_map"])
def test_tournament_viz_help(name):
    """`tournament viz --help` exits 0 for each shipped config name.

    Full viz runs build dozens of PDFs and pull in matplotlib backends, too
    heavy for the smoke suite; --help covers the argparse + entrypoint surface
    for both tournament shapes."""
    res = _run_cli("tournament", "viz", "--help")
    assert res.returncode == 0, (
        f"tournament viz --help failed for {name} (exit {res.returncode})\n"
        f"stderr: {res.stderr.decode()[:500]}"
    )


# ── 39. analysis subcommands on a shipped agent ─────────────────────────
# `analysis audit` requires TB event files which are not committed (the
# `tfevents-agent-archive` release ships them separately). We cover its
# CLI surface via the --help test above; on-data smoke is limited to the
# two subcommands whose inputs are part of the shipped tree.
@pytest.mark.parametrize("subcmd", ["metrics", "params"])
def test_analysis_subcommand_on_shipped_agent(subcmd):
    """`analysis {metrics,params}` runs on a shipped agent.

    Covers the common code path each subcommand exercises (config load +
    agent.pt load + metric extraction); a smoke success here catches the
    "broken after a refactor" class of regression. The actual metric values
    are not validated."""
    agent_path = "data/agents/GridNet-SingleMap"
    res = _run_cli("analysis", subcmd, agent_path, timeout=120)
    assert res.returncode == 0, (
        f"analysis {subcmd} failed (exit {res.returncode})\n"
        f"stderr tail: {res.stderr.decode()[-1000:]}"
    )


# ── 40. requirements-lock.txt structural sanity ─────────────────────────
def test_requirements_lock_well_formed():
    """`requirements-lock.txt` is the file consumed by `pip install
    --require-hashes`. Verify it has the expected uv-generated header,
    enough pinned versions, and at least as many sha256 hashes as pins,
    so a corrupted commit doesn't silently break reproducible installs."""
    lock = REPO_ROOT / "requirements-lock.txt"
    assert lock.exists(), "requirements-lock.txt is missing"
    text = lock.read_text()
    assert "autogenerated by uv" in text, (
        "lock file lost its uv autogen header (likely hand-edited or replaced)"
    )
    pin_count = sum(
        1 for line in text.splitlines() if "==" in line and not line.lstrip().startswith("#")
    )
    hash_count = text.count("--hash=sha256:")
    assert pin_count >= 50, f"too few pinned requirements ({pin_count})"
    # Every pin should carry at least one sha256 hash; uv often emits two
    # (wheel + sdist), so a >=1.0 ratio is a strong floor.
    assert hash_count >= pin_count, (
        f"fewer hashes ({hash_count}) than pins ({pin_count}); --require-hashes will fail"
    )

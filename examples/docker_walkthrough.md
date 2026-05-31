# Docker walkthrough

End-to-end recipe for running MicroRTS-DRL agents through the shipped Docker
images. No conda, no JDK install on the host: everything happens inside the
container. Two images:

| Image | Purpose | Base | Size |
|---|---|---|---|
| `Dockerfile` (CPU) | Evaluation, smoke runs, BC training | `python:3.10-slim` | ~2.7 GB |
| `Dockerfile.gpu` | Full training | `nvidia/cuda:12.9.2-cudnn-runtime-ubuntu22.04` | ~6.5 GB |

The CPU image is the default for evaluation. The GPU image needs the
[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
and a CUDA-capable GPU on the host.

## 1. Build the CPU image

```bash
docker build -t microrts-drl-uecd .
```

This:
1. Pulls `python:3.10-slim` + installs OpenJDK 17 (for the JNI bridge).
2. Installs CPU-only torch + the project deps from `pyproject.toml`.
3. Fetches the RAISocketAI competition wheel (~225 MB) from the GitHub
   release. Skip it with `--build-arg SKIP_RAISOCKETAI=1` if you don't
   need to face RAISocketAI in tournaments.
4. Builds the Java bridge against the vendored MicroRTS engine.
5. Runs a smoke `import microrts_agent` to catch a broken image early.

Total build time on a warm cache: 2-3 minutes.

## 2. Print the CLI help

```bash
docker run --rm microrts-drl-uecd
```

The image's `ENTRYPOINT` is `microrts-agent`, so passing no arguments prints
the help screen. To run any subcommand, append its name and flags:

```bash
docker run --rm microrts-drl-uecd train --help
docker run --rm microrts-drl-uecd evaluate --help
docker run --rm microrts-drl-uecd tournament --help
```

## 3. Evaluate a shipped agent (out of the box)

The image bakes in `data/agents/` (the 9 shipped trained agents), so the
following works with no host mount:

```bash
docker run --rm microrts-drl-uecd evaluate \
    --agent data/agents/UECD-SingleMap-Best \
    --opponent CoacAI \
    --maps maps/open_competition/basesWorkers16x16A.xml \
    --nb_games 10 \
    --max-steps 4000
```

Expected: `UECD-SingleMap-Best` wins ~95% of 10 games against `CoacAI` on
the canonical 16x16 map. See `data/tournaments/single_map/` for the full
19-agent round-robin numbers.

## 4. Mount extra data subtrees

Only `data/agents/` is baked in. Other `data/` subtrees (`ablation/`, `BC/`,
`tournaments/`, `recordings/`, `rush_collapse/`, `generalization_probes/`)
are dropped by `.dockerignore` to keep the image small. Mount them as
volumes when you need them.

**BC training** (needs the teacher dataset under `data/BC/training/`):

```bash
docker run --rm -v "$(pwd)/data/BC:/app/data/BC" microrts-drl-uecd \
    bc train \
    --data 'data/BC/training/bc_chunk_*.npz' \
    --architecture unet_entity_cbam_deep \
    --epochs 5 \
    --batch-size 128 \
    --output outputs/runs/docker-bc-smoke
```

**Run a tournament** (needs `tournament_configs/` shipped + extra agents):

```bash
docker run --rm -v "$(pwd)/data:/app/data" microrts-drl-uecd \
    tournament run single_map
```

**Persist training output** (so the host can see what was written):

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" microrts-drl-uecd \
    train --exp-name docker-smoke --total-timesteps 100000
```

## 5. GPU image (training)

Build:

```bash
docker build -f Dockerfile.gpu -t microrts-drl-uecd:gpu .
```

Run (the `--gpus all` flag wires the GPU into the container):

```bash
docker run --rm --gpus all microrts-drl-uecd:gpu                 # help
docker run --rm --gpus all microrts-drl-uecd:gpu evaluate \
    --agent data/agents/UECD-SingleMap-Best \
    --opponent CoacAI \
    --maps maps/open_competition/basesWorkers16x16A.xml \
    --nb_games 10
docker run --rm --gpus all -v "$(pwd)/outputs:/app/outputs" \
    microrts-drl-uecd:gpu train --exp-name docker-gpu-smoke \
    --total-timesteps 100000
```

Inside the container, `torch.cuda.is_available()` should report `True`.
A full thesis-scale single-map run (350 M steps on `basesWorkers16x16A`)
takes ~10 GPU-days on an A100, well beyond a quick smoke test.

## 6. Common pitfalls

- **"file not found" inside the container**: the working directory is
  `/app/`. Paths like `data/agents/UECD-SingleMap-Best` resolve relative
  to that, not to your host pwd.
- **Permission errors on bind mounts**: the container runs as root by
  default. Files it writes to a host-mounted volume are owned by uid 0.
  If you need a non-root user, build with `--build-arg USER_UID=$(id -u)`
  (not implemented by default; see the Dockerfile header note).
- **`--gpus all` fails with "could not select device driver"**: the
  NVIDIA Container Toolkit isn't installed on the host. See its
  [setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **CPU build fails on `SKIP_RAISOCKETAI=0`**: outbound HTTPS to
  `github.com` is required to fetch the wheel; either fix the network or
  rebuild with `--build-arg SKIP_RAISOCKETAI=1`.

## Where to look next

- `examples/load_agent.ipynb`: same eval as Section 3 above, but from
  inside a Jupyter notebook (CPU).
- `examples/showcase_results.ipynb`: head-to-head matrix across the
  shipped agents and the full opponent pool.
- `examples/tournament_walkthrough.ipynb`: small reproducible tournament,
  parsed + plotted.
- `README.md` Docker section: the canonical build / run snippets that
  this walkthrough expands on.

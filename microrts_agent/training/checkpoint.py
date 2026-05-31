"""Checkpoint save/load, device setup, TensorBoard init, stdout Tee."""

import os
import random
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def setup_device_and_seed(args):
    """CUDA if available, seed all RNGs, enable deterministic cuDNN."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    return device


def _atomic_save(obj, path):
    """Write to .tmp then rename: survives SLURM job kills."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(agent, optimizer, global_step, update, run_dir):
    """Save agent.pt (weights), {step}.pt (snapshot), checkpoint.pt (resumable)."""
    weights = agent.state_dict()
    full_ckpt = {
        "model": weights,
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "update": update,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    _atomic_save(weights, os.path.join(run_dir, "agent.pt"))
    _atomic_save(full_ckpt, os.path.join(run_dir, f"{global_step}.pt"))
    _atomic_save(full_ckpt, os.path.join(run_dir, "checkpoint.pt"))


def setup_tensorboard(args, run_dir):
    """Create SummaryWriter and log all hyperparameters as a markdown table."""
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )
    return writer


def resume_checkpoint(args, agent, optimizer, device):
    """Restore model, optimizer, step counter, and RNG states from --resume.
    Returns (starting_update, global_step).
    """
    starting_update = 1
    global_step = 0

    if args.resume:
        ckpt_path = os.path.join(args.resume, "checkpoint.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt["global_step"]
            starting_update = ckpt["update"] + 1
            try:
                torch.set_rng_state(ckpt["rng_torch"])
                np.random.set_state(ckpt["rng_numpy"])
                random.setstate(ckpt["rng_python"])
            except (TypeError, KeyError) as e:
                print(f"  WARNING: could not restore RNG state ({e}), continuing with fresh RNG")
            print(
                f"Resumed from {ckpt_path} (update={ckpt['update']}, global_step={global_step:,})"
            )
        else:
            print(f"ERROR: checkpoint.pt not found in {args.resume}")
            sys.exit(1)

    return starting_update, global_step


def _open_log_file(filepath):
    """Open the Tee log file (line-buffered). The Tee instance owns the handle and closes it."""
    return open(filepath, "a", buffering=1)


class Tee:
    """Duplicate stdout to both terminal and a line-buffered log file."""

    def __init__(self, filepath):
        self.file = _open_log_file(filepath)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()

    def __getattr__(self, name):
        # Proxy unknown attrs (encoding, fileno, isatty, ...) to the wrapped stdout
        # so libraries that introspect sys.stdout keep working under tee'd output.
        try:
            stdout = self.__dict__["stdout"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(stdout, name)

    def __del__(self):
        self.close()

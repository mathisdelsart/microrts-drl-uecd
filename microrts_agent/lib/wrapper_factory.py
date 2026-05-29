"""Wrapper factory. Applies the canonical wrapper stack from CLI flags.
Order: StatsRecorder → FrameStack → ReservedObs → SymmetryAugmentation.
VecMonitor is applied separately in train.py after this.
"""

from microrts_agent.lib.wrappers.frame_stack import FrameStack
from microrts_agent.lib.wrappers.reserved_obs import ReservedPositionObs
from microrts_agent.lib.wrappers.stats_recorder import StatsRecorder
from microrts_agent.lib.wrappers.symmetry_augmentation import SymmetryAugmentation


def apply_env_wrappers(
    envs, *, gamma=0.99, frame_stack=0, reserved_obs=False, augment_symmetry=False
):
    """Apply observation wrappers in canonical order.

    Returns the (possibly wrapped) env.
    """
    envs = StatsRecorder(envs, gamma)
    if frame_stack > 1:
        envs = FrameStack(envs, num_stack=frame_stack)
    if reserved_obs:
        envs = ReservedPositionObs(envs)
    if augment_symmetry:
        envs = SymmetryAugmentation(envs)
    return envs

"""Base class for all MicroRTS vectorized environments.

Handles JVM bootstrap (once per process, cannot restart), UnitTypeTable,
render, close. Subclassed by MicroRTSRLVecEnv (RL) and MicroRTSBotVecEnv (tournaments).
"""

import os
from abc import ABC, abstractmethod

import jpype
import numpy as np
from jpype.imports import registerDomain
from PIL import Image

from microrts_agent.paths import MICRORTS_DIR


def get_base_env(env):
    """Unwrap VecEnvWrapper chain to get the underlying MicroRTS env."""
    while hasattr(env, "venv"):
        env = env.venv
    return env


def suppress_java_output():
    """Redirect Java stdout/stderr to /dev/null (once per process).
    Must be called after the JVM is started.
    """
    null_ps = jpype.JClass("java.io.PrintStream")(
        jpype.JClass("java.io.OutputStream").nullOutputStream()
    )
    jpype.JClass("java.lang.System").setOut(null_ps)
    jpype.JClass("java.lang.System").setErr(null_ps)


# ------------------------------------------------------------------
# Base environment class
# ------------------------------------------------------------------


class BaseMicroRTSVecEnv(ABC):
    metadata = {"render_mode": "rgb_array", "render_fps": 150}

    def __init__(self, num_envs, partial_obs, max_steps, map_paths, jvm_args):

        self.num_envs = num_envs
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.map_paths = map_paths
        self.microrts_path = str(MICRORTS_DIR)
        self.render_mode = "rgb_array"

        # JARs to add to the classpath (relative to microrts/)
        # bridge.jar FIRST: our compiled src/ overrides classes in microrts.jar
        _JARS = [
            "lib/bridge.jar",
            "microrts.jar",
            "lib/bots/Coac.jar",
            "lib/bots/Droplet.jar",
            "lib/bots/mayariBot.jar",
            "lib/bots/MixedBot.jar",
            "lib/bots/ObiBotKenobi.jar",
            "lib/bots/RAISocketAI.jar",
            "lib/bots/StrategyTactics.jar",
            "lib/bots/TiamatBot.jar",
            "lib/bots/TMA.jar",
            "lib/bots/UTS_Imass.jar",
        ]

        # Launch JVM once per process (cannot be restarted after shutdown)
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            registerDomain("rts")
            registerDomain("standard")
            registerDomain("tma")

            microrts_path = str(MICRORTS_DIR)
            for jar in _JARS:
                jpype.addClassPath(os.path.join(microrts_path, jar))

            if jvm_args is None:
                jvm_args = []
            jpype.startJVM(*jvm_args, convertStrings=False)

        # UnitTypeTable: needed by every env instance (not just the first)
        from rts.units import UnitTypeTable  # type: ignore[import]

        self.real_utt = UnitTypeTable()

    # ------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def start_client(self):
        """Create the Java JNIGridnetVecClient and instantiate bot AIs.
        Called once in __init__ after reward functions are set up."""

    @abstractmethod
    def reset(self):
        """Reset all parallel games. Returns observations (B, H, W, C) or None for bot envs."""

    @abstractmethod
    def step_async(self, actions):
        """Convert RL actions to Java format (prepend cell idx, filter to cells with units).
        No-op for bot envs (bots compute their own actions in Java)."""

    @abstractmethod
    def step_wait(self):
        """Execute one game tick in Java, encode obs, compute reward.
        Returns (obs, reward, done, infos)."""

    # ------------------------------------------------------------------
    # Gym VecEnv interface (shared)
    # ------------------------------------------------------------------

    def step(self, actions):
        """Full step: prepare actions then execute. Returns (obs, reward, done, infos)."""
        self.step_async(actions)
        return self.step_wait()

    # ------------------------------------------------------------------
    # Rendering & cleanup
    # ------------------------------------------------------------------

    def get_attr(self, attr_name, indices=None):
        """Return attribute value(s), required by SB3 VecEnv interface."""
        val = getattr(self, attr_name)
        if indices is None:
            return [val] * self.num_envs
        return [val] * len(indices)

    def render(self, mode="rgb_array"):
        bytes_array = np.array(self.render_client.render(True))
        # Java returns TYPE_3BYTE_BGR → PIL misinterprets as RGB → flip to correct
        image = Image.frombytes("RGB", (640, 640), bytes_array)
        return np.ascontiguousarray(np.array(image)[:, :, ::-1])

    def close(self):
        """Shut down Java client (GUI windows, sockets).

        Does NOT shut down the JVM: JPype cannot restart it, so other env
        instances (eval, selfplay) would become unusable.  The JVM is cleaned
        up automatically when the process exits.
        """
        if jpype._jpype.isStarted() and hasattr(self, "vec_client"):
            self.vec_client.close()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def getattr_depth_check(self, name, already_found):
        """
        Check if an attribute reference is being hidden in a recursive call to __getattr__
        :param name: (str) name of attribute to check for
        :param already_found: (bool) whether this attribute has already been found in a wrapper
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        if hasattr(self, name) and already_found:
            return f"{type(self).__module__}.{type(self).__name__}"
        else:
            return None

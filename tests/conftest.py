"""Shared pytest fixtures.

The JVM is started once per session because JPype cannot stop and
restart it inside the same process; every test that needs Java work
shares the same instance.
"""

import os

import pytest

_JARS = [
    "lib/bridge.jar",  # FIRST: our compiled src/ overrides classes in microrts.jar
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


@pytest.fixture(scope="session")
def jvm():
    """Start the JVM once per session with the full microrts classpath.

    Mirrors the BaseVecEnv classpath setup so tests can reach Java
    classes without instantiating a full vec env first.
    """
    import jpype
    from jpype.imports import registerDomain

    from microrts_agent.envs.factory import JVM_ARGS
    from microrts_agent.paths import MICRORTS_DIR

    if not jpype.isJVMStarted():
        registerDomain("ts", alias="tests")
        registerDomain("ai")
        registerDomain("rts")
        registerDomain("standard")
        registerDomain("tma")
        for jar in _JARS:
            jpype.addClassPath(os.path.join(str(MICRORTS_DIR), jar))
        jpype.startJVM(*JVM_ARGS, convertStrings=False)
    return jpype

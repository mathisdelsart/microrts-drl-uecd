# NOTE: All `from ai.*` / `from GNS` / `from mayariBot` / etc. imports below
# are JPype Java domain imports (resolved at runtime via the JVM classpath),
# NOT Python modules. Pylance warnings about missing imports are expected.


def randomBiasedAI(utt):
    from ai import RandomBiasedAI  # type: ignore[import-not-found]

    return RandomBiasedAI()


def randomAI(utt):
    from ai import RandomBiasedSingleUnitAI  # type: ignore[import-not-found]

    return RandomBiasedSingleUnitAI()


def passiveAI(utt):
    from ai import PassiveAI  # type: ignore[import-not-found]

    return PassiveAI()


def workerRushAI(utt):
    from ai.abstraction import WorkerRush  # type: ignore[import-not-found]

    return WorkerRush(utt)


def lightRushAI(utt):
    from ai.abstraction import LightRush  # type: ignore[import-not-found]

    return LightRush(utt)


def POLightRush(utt):
    from ai.abstraction.partialobservability import POLightRush  # type: ignore[import-not-found]

    return POLightRush(utt)


def POWorkerRush(utt):
    from ai.abstraction.partialobservability import POWorkerRush  # type: ignore[import-not-found]

    return POWorkerRush(utt)


def POHeavyRush(utt):
    from ai.abstraction.partialobservability import POHeavyRush  # type: ignore[import-not-found]

    return POHeavyRush(utt)


def PORangedRush(utt):
    from ai.abstraction.partialobservability import PORangedRush  # type: ignore[import-not-found]

    return PORangedRush(utt)


def naiveMCTSAI(utt):
    from ai.mcts.naivemcts import NaiveMCTS  # type: ignore[import-not-found]

    return NaiveMCTS(utt)


# https://github.com/AmoyZhp/MixedBotmRTS
def mixedBot(utt):
    from ai.JZ import MixedBot  # type: ignore[import-not-found]

    return MixedBot(utt)


# https://github.com/jr9Hernandez/TiamatBot
def tiamat(utt):
    from ai.competition.tiamat import Tiamat  # type: ignore[import-not-found]

    return Tiamat(utt)


# https://github.com/zuozhiyang/Droplet/blob/master/GNS/Droplet.java
def droplet(utt):
    from GNS import Droplet  # type: ignore[import-not-found]

    return Droplet(utt)


# https://github.com/barvazkrav/mayariBot/blob/master/mayari.java
def mayari(utt):
    from mayariBot import mayari  # type: ignore[import-not-found]

    return mayari(utt)


# https://github.com/Coac/coac-ai-microrts/tree/master
def coacAI(utt):
    from ai.coac import CoacAI  # type: ignore[import-not-found]

    return CoacAI(utt)


def obiBotKenobi(utt):
    import jpype

    return jpype.JClass("ObiBotKenobi")(utt)


def strategyTactics(utt):
    from standard import StrategyTactics  # type: ignore[import-not-found]

    return StrategyTactics(utt)


def tmaAI(utt):
    from tma import TMA  # type: ignore[import-not-found]

    return TMA(utt)


def raiSocketAI(utt):
    import jpype

    return jpype.JClass("ai.rai.RAISocketAI")(utt)


def utsImass(utt):
    import jpype

    return jpype.JClass("ai.socket.UTS_Imass")(utt)


AI_MAPPING = {
    "RandomAI": randomAI,
    "RandomBiasedAI": randomBiasedAI,
    "PassiveAI": passiveAI,
    "WorkerRushAI": workerRushAI,
    "LightRushAI": lightRushAI,
    "CoacAI": coacAI,
    "Mayari": mayari,
    "NaiveMCTS": naiveMCTSAI,
    "MixedBot": mixedBot,
    "Tiamat": tiamat,
    "Droplet": droplet,
    "LightRush": POLightRush,
    "WorkerRush": POWorkerRush,
    "POHeavyRush": POHeavyRush,
    "PORangedRush": PORangedRush,
    "ObiBotKenobi": obiBotKenobi,
    "StrategyTactics": strategyTactics,
    "TMA": tmaAI,
    "RAISocketAI": raiSocketAI,
    "UtsImass": utsImass,
}

# Bots used in the thesis tournament (subset of AI_MAPPING)
TOURNAMENT_BOT_NAMES = [
    "CoacAI",
    "Mayari",
    "Tiamat",
    "MixedBot",
    "ObiBotKenobi",
    "StrategyTactics",
    "TMA",
    "RandomBiasedAI",
    "LightRush",
    "WorkerRush",
    "RAISocketAI",
    "UtsImass",
]
TOURNAMENT_AI_MAPPING = {name: AI_MAPPING[name] for name in TOURNAMENT_BOT_NAMES}

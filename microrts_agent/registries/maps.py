# ============================================================================
# MicroRTS Map Registry
# All paths relative to microrts_agent/microrts/ (prefix "maps/")
# ============================================================================

# ── Competition ──────────────────────────────────────────────────────────────

COMPETITION_OPEN_MAPS = [
    "maps/open_competition/basesWorkers8x8A.xml",
    "maps/open_competition/FourBasesWorkers8x8.xml",
    "maps/open_competition/NoWhereToRun9x8.xml",
    "maps/open_competition/basesWorkers16x16A.xml",
    "maps/open_competition/TwoBasesBarracks16x16.xml",
    "maps/open_competition/DoubleGame24x24.xml",
    "maps/open_competition/BWDistantResources32x32.xml",
    "maps/open_competition/BloodBath.scmB.xml",  # 64x64
]

COMPETITION_HIDDEN_MAPS = [
    "maps/closed_competition/chambers32x32.xml",
    "maps/closed_competition/itsNotSafe.xml",  # 15x14
    "maps/closed_competition/basesWorkers32x32A.xml",
    "maps/closed_competition/basesWorkers24x24A.xml",
]

# ── Subset pools ────────────────────────────────────────────────────────
# Maps ≤ 16×16
COMPETITION_OPEN_SMALL = [
    m for m in COMPETITION_OPEN_MAPS if any(tag in m for tag in ("8x8", "9x8", "16x16"))
]

# ── Per-map metadata ────────────────────────────────────────────────────
# Competition-standard max game cycles per map.
# Source: MicroRTS competition rules & map XML conventions.
MAP_MAX_CYCLES = {
    # Open maps
    "maps/open_competition/basesWorkers8x8A.xml": 3_000,
    "maps/open_competition/FourBasesWorkers8x8.xml": 3_000,
    "maps/open_competition/NoWhereToRun9x8.xml": 4_000,
    "maps/open_competition/basesWorkers16x16A.xml": 4_000,
    "maps/open_competition/TwoBasesBarracks16x16.xml": 4_000,
    "maps/open_competition/DoubleGame24x24.xml": 5_000,
    "maps/open_competition/BWDistantResources32x32.xml": 6_000,
    "maps/open_competition/BloodBath.scmB.xml": 8_000,
    # Closed maps
    "maps/closed_competition/itsNotSafe.xml": 4_000,
    "maps/closed_competition/basesWorkers24x24A.xml": 5_000,
    "maps/closed_competition/basesWorkers32x32A.xml": 6_000,
    "maps/closed_competition/chambers32x32.xml": 6_000,
}


def get_max_cycles(map_path, default=4_000):
    """Return competition-standard max game cycles for a map path.
    Falls back to `default` for unknown maps.
    """
    return MAP_MAX_CYCLES.get(map_path, default)


def map_short(path):
    """'maps/open_competition/basesWorkers16x16A.xml' -> 'basesWorkers16x16A'"""
    import os

    return os.path.splitext(os.path.basename(path))[0]

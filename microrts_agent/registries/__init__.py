"""
Lookup tables that map string names to Java bots and map paths.

ai.py:
  Each function (e.g. workerRush, coacAI) is a factory that takes a
  UnitTypeTable and returns a Java AI instance via JPype. The imports
  inside each function are Java packages resolved at runtime through
  the JVM classpath: they are NOT Python modules.
  AI_MAPPING maps human-readable names ("CoacAI", "Tiamat", ...) to
  these factories. Used everywhere: train.py, evaluate.py, tournaments.

maps.py:
  Lists of map XML paths (relative to microrts/) grouped by category
  (competition open/hidden). Used to configure which maps to train or
  evaluate on.
"""

"""Optional feature modules for actor-critic architectures.

These are plugged into the base architecture via _finalize() flags:
  - autoregressive.py : --autoregressive (sequential sub-action conditioning)
  - hl_gauss.py       : --hl-gauss (value via classification instead of regression)
  - popart.py         : --popart (adaptive value normalization)
  - auxiliary_heads.py : --aux-tasks {spatial,contrastive,unit_count,opponent_modeling}
                         (encoder enrichment via auxiliary supervised heads)
"""

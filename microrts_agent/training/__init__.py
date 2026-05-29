"""
Training infrastructure for PPO on MicroRTS.

  - cli:          CLI argument definitions, parsing, validation
  - checkpoint:   save/load checkpoints, device setup, TensorBoard init
  - config:       run configuration serialization and banner display
  - opponents:    opponent selection and configuration
  - logging:      TensorBoard logging helpers
  - ppo:          GAE computation and PPO policy gradient update
  - scheduling:   OpponentTracker, CheckpointPool, reward scheduling
  - selfplay:     SelfPlayManager (opponent model, pool, side alternation)
  - eval:         in-training evaluation against bot pool
  - auxiliary:    auxiliary task heads and loss computation
  - setup:        eval env setup, aux helpers, GAE dispatch
"""

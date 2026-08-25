# Curated TRIGGERED critic-fix rollout clips

Four representative episodes pulled from the 30-episode, seed-10000 speed-comparison
set (`results/speed_comparison/trained_criticfix_30ep_seed10000_run2_withvideo.json`,
`results/speed_comparison/videos/`), covering the success case and each of the three
failure modes described in the top-level README's failure-mode taxonomy. Selected by
cross-referencing each episode's logged `trigger_active_fraction` (how much of the
episode the heuristic TRIGGERED gate was active) against its `success` flag.

- `episode_0_success.mp4` — a clean success (trigger active 14.6% of the episode).
- `episode_16_failure.mp4` — **never-triggered** failure: trigger active 0% of the
  episode. The gripper never gets both close enough to the peg and slow enough to
  enter the critical-phase window within the horizon, so the learned correction
  never gets a chance to act. A reach/transport-phase failure, not a
  last-millimeter one.
- `episode_17_failure.mp4` — **partial-engagement** failure: trigger active 36.75%
  of the episode (the taxonomy's 24-37% band). The correction gets a chance to act
  but the episode still fails.
- `episode_29_failure.mp4` — **stuck-in-window** failure: trigger active 80.75% of
  the episode (the taxonomy's 73-81% band, the most extreme example in this set).
  The gripper spends most of the episode inside the critical-phase zone without
  completing the insertion.

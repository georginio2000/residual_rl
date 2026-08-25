# All 30 episodes, concatenated

`rollouts_all30.mp4` — every episode from the 30-episode, seed-10000 speed-comparison
set (`results/speed_comparison/trained_criticfix_30ep_seed10000_run2_withvideo.json`),
concatenated in original episode order via stream copy (lossless, no re-encode) from
the per-episode clips in `results/speed_comparison/videos/`. This mirrors how
`results/bc_rnn_baseline/videos/rollouts.mp4` was produced for the frozen BC-RNN
baseline (robomimic's `run_trained_agent.py` concatenates every evaluated rollout
into one file by default) — the TRIGGERED critic-fix policy has no robomimic
equivalent to run that script against, since it's the frozen BC-RNN composed with a
trained SAC residual correction at inference time, routed through the project's own
`scripts/robomimic/square_bridge_service.py --trigger` bridge instead. Concatenating
that bridge's own per-episode outputs is the equivalent for this policy.

512x256, 20fps, 6182 frames, 309.1s total. 23/30 episodes succeed (76.7%), matching
`speed_comparison.png`'s frozen-vs-trained comparison sample.

| Episode | Start | End | Result | Trigger-active fraction |
|---:|---:|---:|:---:|---:|
| 0 | 0:00 | 0:07 | success | 0.146 |
| 1 | 0:07 | 0:14 | success | 0.156 |
| 2 | 0:14 | 0:34 | **failure** | 0.730 (stuck-in-window) |
| 3 | 0:34 | 0:41 | success | 0.171 |
| 4 | 0:41 | 0:48 | success | 0.336 |
| 5 | 0:48 | 0:55 | success | 0.358 |
| 6 | 0:55 | 1:06 | success | 0.458 |
| 7 | 1:06 | 1:13 | success | 0.254 |
| 8 | 1:13 | 1:20 | success | 0.267 |
| 9 | 1:20 | 1:27 | success | 0.183 |
| 10 | 1:27 | 1:34 | success | 0.173 |
| 11 | 1:34 | 1:41 | success | 0.278 |
| 12 | 1:41 | 1:50 | success | 0.281 |
| 13 | 1:50 | 2:10 | **failure** | 0.245 (partial-engagement) |
| 14 | 2:10 | 2:16 | success | 0.171 |
| 15 | 2:16 | 2:23 | success | 0.290 |
| 16 | 2:23 | 2:43 | **failure** | 0.000 (never-triggered) |
| 17 | 2:43 | 3:03 | **failure** | 0.367 (partial-engagement) |
| 18 | 3:03 | 3:13 | success | 0.219 |
| 19 | 3:13 | 3:19 | success | 0.154 |
| 20 | 3:19 | 3:39 | **failure** | 0.343 (partial-engagement) |
| 21 | 3:39 | 3:46 | success | 0.191 |
| 22 | 3:46 | 3:54 | success | 0.264 |
| 23 | 3:54 | 4:01 | success | 0.170 |
| 24 | 4:01 | 4:08 | success | 0.150 |
| 25 | 4:08 | 4:15 | success | 0.320 |
| 26 | 4:15 | 4:22 | success | 0.209 |
| 27 | 4:22 | 4:29 | success | 0.127 |
| 28 | 4:29 | 4:49 | **failure** | 0.752 (stuck-in-window) |
| 29 | 4:49 | 5:09 | **failure** | 0.807 (stuck-in-window) |

For a shorter, purpose-built cut instead, see `../videos_curated/` (one success plus
one example of each of the three failure modes above).

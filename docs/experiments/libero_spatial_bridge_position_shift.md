# Bridge-Level Frozen π0.5 LIBERO-Spatial Position Shift

## Objective

Confirm that the +14 cm world-x scene shift selected with the direct LIBERO
runner remains a mid-success deployment condition when evaluated through the
normal `residual_rl` entry point and isolated state/action bridge. No residual
policy was trained and no VLA latent was extracted.

## Frozen and isolated components

- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO submodule: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- OpenPI inference batch size: 1
- OpenPI server: official Docker Compose service
- LIBERO client: separate Python 3.8/Torch 1.11 `libero` container
- Main evaluator: Python 3.10 `last_millimeter.evaluate`
- Policy and action calibration: frozen and unmodified

The main environment fixes `scene_translation` in configuration and sends it
on each `/reset`. The bridge validates the request, loads the official initial
state, translates every movable object and fixture, verifies the achieved
root positions, performs ten official stabilization actions, and only then
requests the first frozen action. Per-reset overrides are rejected, and a
nonzero scene translation cannot be combined with action bias.

## Scenario split

The pinned Spatial suite contains 50 official initial states per task. The
tracked split is `configs/libero/splits/spatial_v1.yaml`:

| Partition | State IDs | Purpose |
|:---|:---|:---|
| Selection | 0–2 | Chose and confirmed the +14 cm perturbation |
| Train | 3–39 | Future residual-policy training only |
| Validation | 40–44 | Model selection and early stopping |
| Test | 45–49 | Final report after all choices are frozen |

The four partitions are disjoint and cover all 50 states. Because states 0–2
were used to select the perturbation, the results below are premise-selection
evidence, not final held-out performance.

## Commands

Start the pinned server and the ten-task bridge as described in the README.
The bridge does not need a perturbation flag; the main-project environment owns
that fixed experimental condition. Run each condition after restarting the
frozen server and launching a fresh bridge artifact directory:

```bash
.venv/bin/python -m last_millimeter.evaluate \
  --config configs/libero/spatial_suite_scene_shift_x_0p000_frozen_state.yaml \
  --episodes 30

.venv/bin/python -m last_millimeter.evaluate \
  --config configs/libero/spatial_suite_scene_shift_x_0p140_frozen_state.yaml \
  --episodes 30
```

Both configs use round-robin scheduling over all ten tasks and initial-state
IDs 0–2, producing the same 30 scenario IDs once per condition.

## Results

The bridge-level evaluation ran on 2026-08-21:

| Condition | Successes | Success rate | 95% Wilson interval |
|:---|---:|---:|:---|
| Zero-shift frozen control | 29/30 | 96.7% | 83.3–99.4% |
| +14 cm frozen scene shift | 18/30 | 60.0% | 42.3–75.4% |

The shift caused 11 fewer successes and a 36.7 percentage-point decrease. Each
pattern below is ordered by initial states 0, 1, and 2 (`S` = success, `F` =
timeout/failure):

| Task | Short description | Zero shift | +14 cm | Shifted successes |
|---:|:---|:---:|:---:|---:|
| 0 | bowl between plate and ramekin | `SSS` | `SSS` | 3/3 |
| 1 | bowl next to ramekin | `SSS` | `FSS` | 2/3 |
| 2 | bowl at table center | `SSS` | `FFF` | 0/3 |
| 3 | bowl on cookie box | `SSS` | `SSS` | 3/3 |
| 4 | bowl in top drawer | `SSS` | `FFF` | 0/3 |
| 5 | bowl on ramekin | `SSS` | `SSS` | 3/3 |
| 6 | bowl next to cookie box | `SSS` | `FFF` | 0/3 |
| 7 | bowl on stove | `SSS` | `SFS` | 2/3 |
| 8 | bowl next to plate | `SSS` | `SSF` | 2/3 |
| 9 | bowl on cabinet | `SFS` | `SSS` | 3/3 |

Across all 60 episodes, maximum root-translation error was exactly 0 m. All
30 zero-shift nominal/post-translation hashes matched, while all 30 +14 cm
hash pairs differed. A post-evaluation memory sample reported 9,084 MiB used
and 2,827 MiB free of 12,288 MiB; the established active-rollout peak remains
9,622 MiB.

Raw JSON and videos remain in ignored local directories:

- `runs/libero_bridge_spatial_scene_shift_x_0p000_selection/`
- `runs/libero_bridge_spatial_scene_shift_x_0p140_selection/`

A compact tracked summary is stored in
`docs/experiments/libero_spatial_bridge_position_shift_results.json`.

## Interpretation

The normal project boundary reproduces the desired deployment regime: the
frozen generalist is near ceiling without the shift and at 60% with it. The
direct runner previously measured 53.3% under +14 cm; that small difference is
consistent with known accelerator-level policy stochasticity, and both results
fall in the predeclared 50–70% target band.

This completes the frozen-baseline gate. It does not establish residual-RL
recovery. Any later training must use only states 3–39, tune on states 40–44,
and leave states 45–49 untouched until the final evaluation.

# Frozen π0.5 LIBERO-Spatial Position-Shift Sweep

## Research question

Does frozen `pi05_libero` performance degrade when a valid LIBERO-Spatial
scene is moved outside its nominal workspace distribution while task semantics
and controller calibration remain unchanged?

## Frozen components

- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- Policy checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- LIBERO submodule: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Inference batch size: 1
- Replan interval: 5 actions
- Seed: 7
- Action calibration bias: 0

## Perturbation

The single swept parameter is a positive world-x translation, in meters. After
loading an official LIBERO initial state and before the ten stabilization
steps, the runner translates:

- every movable object's 7-DoF free joint; and
- every fixed fixture's root body.

All roots receive the same `[shift_x, 0, 0]` translation. This preserves
relations such as “between,” “on the stove,” and “in the drawer,” as well as
the relative pose between the bowl and target plate. The robot, cameras,
language instruction, controller, and policy are unchanged. Consequently, the
experiment measures a workspace-position distribution shift rather than a
changed task or simultaneous action perturbation.

The implementation rejects non-finite shifts, shifts larger than 20 cm, and
simultaneous action bias. It verifies every requested root translation to
within 1 micrometer. Each episode records nominal, shifted, stabilized, and
final simulator-state hashes; nominal, shifted, and stabilized root positions;
raw and executed action hashes; and a replay video.

## Protocol

The coarse sweep evaluates initial state 0 for each of the ten Spatial tasks.
The frozen server is restarted before each stratum to reset its policy RNG as
consistently as possible. Any candidate transition region is then evaluated on
additional paired initial states before selecting a training perturbation.

The command below runs one stratum inside the isolated Python 3.8 LIBERO
client, against an already-ready frozen OpenPI server:

```bash
mkdir -p /root/workspace/residual_rl/runs/libero_spatial_scene_shift_x_0p100_state0

docker run --rm --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/workspace/openpi:/app:ro \
  -v /root/workspace/residual_rl:/residual_rl:ro \
  -v /root/workspace/residual_rl/runs/libero_spatial_scene_shift_x_0p100_state0:/data \
  libero /bin/bash -lc \
  'source /.venv/bin/activate && \
   python /residual_rl/scripts/openpi/screen_libero_tasks.py \
     --host host.docker.internal \
     --task-suite-name libero_spatial \
     --task-ids 0,1,2,3,4,5,6,7,8,9 \
     --num-trials 1 \
     --initial-state-offset 0 \
     --scene-shift-x 0.10 \
     --action-bias-x 0.0 \
     --seed 7 \
     --output-dir /data'
```

## Results

The experiment ran on 2026-08-21. The state-0 coarse sweep found a sharp
transition between +10 cm and +14 cm:

| World-x scene shift | Successes | Success rate | Failed task IDs |
|---:|---:|---:|:---|
| 0 cm | 10/10 | 100% | none |
| +3 cm | 10/10 | 100% | none |
| +6 cm | 9/10 | 90% | 4 |
| +10 cm | 9/10 | 90% | 4 |
| +14 cm | 5/10 | 50% | 1, 2, 4, 6, 7 |

The candidate +14 cm shift and its zero-shift control were then evaluated on
initial states 0, 1, and 2 for every task. Both used the current instrumented
runner and a freshly restarted server for each condition:

| Condition | Successes | Success rate | 95% Wilson interval |
|:---|---:|---:|:---|
| Zero-shift frozen control | 29/30 | 96.7% | 83.3–99.4% |
| +14 cm frozen scene shift | 16/30 | 53.3% | 36.1–69.8% |

This is a 43.4 percentage-point decrease and 13 fewer successes. The outcome
by task and initial state is below; each three-character pattern is ordered as
states 0, 1, and 2 (`S` = success, `F` = timeout/failure).

| Task | Short description | Zero shift | +14 cm | Shifted successes |
|---:|:---|:---:|:---:|---:|
| 0 | bowl between plate and ramekin | `SSS` | `SSS` | 3/3 |
| 1 | bowl next to ramekin | `SSS` | `FFS` | 1/3 |
| 2 | bowl at table center | `SSS` | `FSF` | 1/3 |
| 3 | bowl on cookie box | `SSS` | `SSS` | 3/3 |
| 4 | bowl in top drawer | `SSS` | `FFF` | 0/3 |
| 5 | bowl on ramekin | `SSS` | `SSF` | 2/3 |
| 6 | bowl next to cookie box | `SSS` | `FFF` | 0/3 |
| 7 | bowl on stove | `SSS` | `FSS` | 2/3 |
| 8 | bowl next to plate | `SSS` | `SFF` | 1/3 |
| 9 | bowl on cabinet | `SFS` | `SSS` | 3/3 |

All 90 distinct instrumented episodes in the sweep and confirmation runs had
zero measured root-translation error. In all 30 zero-shift episodes, the
nominal and post-translation simulator-state hashes were identical. During an
active +10 cm rollout, `nvidia-smi` reported 9,622 MiB used and 2,289 MiB free
of 12,288 MiB, so batch-size-1 frozen inference remains viable on the RTX 3060.

The raw JSON and replay videos are retained locally under these ignored run
directories:

- `runs/libero_spatial_scene_shift_x_0p000_state0_verified/`
- `runs/libero_spatial_scene_shift_x_0p000_states1_2_verified/`
- `runs/libero_spatial_scene_shift_x_0p030_state0/`
- `runs/libero_spatial_scene_shift_x_0p060_state0/`
- `runs/libero_spatial_scene_shift_x_0p100_state0/`
- `runs/libero_spatial_scene_shift_x_0p140_state0/`
- `runs/libero_spatial_scene_shift_x_0p140_states1_2/`

A compact, tracked machine-readable summary is stored in
`docs/experiments/libero_spatial_position_shift_results.json`.

Frozen π0.5 inference exhibits accelerator-level stochastic differences, as
illustrated by the single nominal failure in this run. The two 30-episode
conditions therefore match task and initial-state IDs but are not bitwise
paired trajectories.

## Conclusion

The premise-screening experiment succeeded: a semantics-preserving workspace
shift exists where the frozen generalist falls from near-ceiling performance
to the desired 50–70% band. The +14 cm world-x translation is the candidate
environment shift for a later residual-learning study. This result does **not**
yet show that residual RL can recover the lost performance; no residual LIBERO
training was run.

The subsequent normal-project bridge confirmation measured 29/30 at zero shift
and 18/30 (60.0%) at +14 cm. Its protocol and results are documented in
`docs/experiments/libero_spatial_bridge_position_shift.md`.

## Interpretation and limitations

This perturbation changes the whole semantic scene relative to the fixed robot,
not one object's relation to another. It therefore tests workspace
generalization while avoiding invalid instructions. It does not isolate which
visual feature or reachability margin causes any failure, and one direction of
translation cannot establish rotational or axis symmetry. Three states per
task are sufficient to locate a candidate regime, not to estimate a
publication-grade task success rate. Those are follow-up ablations, not part
of this one-parameter sweep.

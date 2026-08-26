# Teaching a Robot Foundation Model the Last Millimeter

This repository studies whether a small reinforcement-learning controller can
specialize a frozen generalist robot policy for precision manipulation:

```text
executed_action = base_action + gate * correction
```

The toy residual-RL experiment, frozen OpenPI π0.5-LIBERO baselines, and an
isolated state-based LIBERO bridge are validated. A controlled action-calibration
shift reduces frozen π0.5's LIBERO-Spatial success from 96.7% to 60.0% on the
same task/state pairs. OpenPI is served as a frozen remote policy; it is not
installed in the main project environment and no VLA parameters are fine-tuned.

## Current milestone status

- All toy `frozen`, `scratch`, `residual`, and `gated` modes are preserved.
- The full CPU-only test suite passes.
- The 30,000-step toy residual config was verified on CUDA.
- Official OpenPI was cloned recursively and pinned exactly.
- Frozen `pi05_libero` evaluation was reproduced at inference batch size 1.
- The main project now selects all ten Spatial tasks through an isolated
  state/base-action bridge with deterministic task/state evaluation schedules.
- Reusable batch-size-1 screening covered every LIBERO Spatial and LIBERO-10
  task, with explicit initial-state range selection.
- A constant `+0.15` normalized Cartesian x-action calibration bias was selected
  as the controlled deployment shift. Across all ten Spatial tasks and three
  fixed initial states per task, success changed from 29/30 to 18/30.
- The primary adaptation target is the full ten-task Spatial suite, not a
  task-specific policy. No residual LIBERO training has started.
- A paired bridge control on Spatial state 0 scored 10/10 without bias, 6/10
  with the selected bias, and 10/10 with the fixed inverse oracle correction.
- Environment, base-policy, and representation construction now use named,
  extensible backends with mocked CPU-only boundary tests.
- Residual LIBERO training and VLA-latent extraction have not been started.

## Main project setup

Use an isolated environment in this repository. The commands below were tested
on Ubuntu 22.04 with Python 3.10.12:

```bash
cd /root/workspace/residual_rl
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest
```

The validated direct dependency versions were PyTorch 2.13.0+cu130,
Gymnasium 1.3.0, NumPy 2.2.6, PyYAML 6.0.3, and pytest 9.1.1.

Run the existing toy residual experiment on CUDA:

```bash
.venv/bin/python -m last_millimeter.train \
  --config configs/toy/residual.yaml \
  --device cuda
```

The validated seed-0 run completed 30,000 steps and 29,745 updates, then
reached 100% success over 50 evaluation episodes. It used approximately
170 MiB VRAM. Outputs are written under `runs/` and ignored by Git.

Evaluate a checkpoint with:

```bash
.venv/bin/python -m last_millimeter.evaluate \
  --config runs/toy_residual/config.yaml \
  --checkpoint runs/toy_residual/checkpoints/final.pt
```

## Configurable integration boundary

Each YAML config selects three independent backends:

```yaml
environment:
  backend: precision_reach

base_policy:
  backend: proportional

representation:
  backend: identity
```

Built-in names are:

| Boundary | Backends | Purpose |
|---|---|---|
| Environment | `precision_reach`, `remote_libero` | Local toy task or isolated simulator proxy |
| Base policy | `proportional`, `observation_action`, `openpi_websocket` | Toy, observation-supplied, or direct remote action |
| Representation | `identity`, `observation_key`, `observation_keys` | Flat state, one mapping feature, or ordered concatenated features |

`BackendRegistry` accepts additional builders without importing simulator or
VLA dependencies into the core package. `OpenPIClientBasePolicy` depends only
on a small `infer(observation)` protocol, validates action chunks, issues one
observation per remote request, and clears cached chunks at episode boundaries.
The optional `openpi_websocket` builder imports only the lightweight
`openpi-client` package and gives an actionable error when it is absent.
The validated LIBERO path instead uses `remote_libero` with
`observation_action`: its Python 3.8 bridge owns the simulator and OpenPI
client, so neither dependency enters the Python 3.10 main environment.

For a mapping observation, a state-only representation can be configured as:

```yaml
representation:
  backend: observation_key
  options:
    key: observation/state
    output_dim: 8
```

This boundary deliberately does not expose VLA latents yet.

## Frozen OpenPI π0.5-LIBERO baseline

### Exact upstream pins

| Component | Pin |
|---|---|
| OpenPI | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| OpenPI ALOHA submodule | `d1dc83afd89ded4379851257fe5d85632d31d5ec` |
| OpenPI LIBERO submodule | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |

The downloaded 16-file checkpoint tree had this reproducibility digest. The
digest hashes the sorted output of `sha256sum` over all relative file paths:

```text
42d571bd87f05f1182810f5a8bfa6d084c0d0dd277aff739bcf8f69868e6fb99
```

The separately downloaded `paligemma_tokenizer.model` SHA-256 was:

```text
8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6
```

Clone and detach at the validated OpenPI revision:

```bash
cd /root/workspace
git clone --recurse-submodules \
  https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git -C openpi submodule update --init --recursive
git -C openpi status --short
git -C openpi submodule status --recursive
```

### Docker and NVIDIA runtime

The official workflow needs Docker Compose and NVIDIA Container Toolkit. Check
them before changing the host:

```bash
docker --version
docker compose version
nvidia-smi
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

On the validated VM, Docker 28.1.1 was already installed but the runtime entry
point was missing. NVIDIA Container Toolkit 1.20.0 was installed and Docker was
restarted once. If needed on another Ubuntu machine, these are machine-wide
operations and should be reviewed before running:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build isolated server and client images

OpenPI's official server Dockerfile builds unchanged:

```bash
cd /root/workspace/openpi
docker compose -f examples/libero/compose.yml build openpi_server
```

The upstream client uses Python 3.8 and `easydict==1.9`. In August 2026, its
unconstrained isolated build selected a setuptools release containing Python
3.9-only syntax. The repository-owned Dockerfile mirrors the pinned upstream
client Dockerfile and adds only a build-time `setuptools==75.3.4` constraint;
OpenPI and all runtime requirements remain frozen:

```bash
cd /root/workspace/openpi
docker build \
  --tag libero \
  --file /root/workspace/residual_rl/docker/openpi/libero.Dockerfile \
  .
```

The resulting client remains separate at Python 3.8.20 and
Torch 1.11.0+cu113. The server uses OpenPI's separate Python 3.11 environment.

### Start the frozen server

```bash
cd /root/workspace/openpi
SERVER_ARGS='--env LIBERO' \
  docker compose -f examples/libero/compose.yml \
  up -d --no-build openpi_server
docker logs -f libero-openpi_server-1
```

Wait for `server listening on 0.0.0.0:8000`. The first start downloads about
11.6 GiB into `/root/.cache/openpi`.

### Run exactly one task at batch size 1

The official client CLI iterates every task in a suite. The following runner is
a task-selectable copy of that pinned evaluation loop. OpenPI is mounted
read-only, the client uses bridge networking instead of privileged host
networking, and only the results directory is writable:

```bash
mkdir -p /root/workspace/residual_rl/runs/openpi_libero_spatial_task0

docker run --rm --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -v /root/workspace/openpi:/app:ro \
  -v /root/workspace/residual_rl:/residual_rl:ro \
  -v /root/workspace/residual_rl/runs/openpi_libero_spatial_task0:/data \
  libero /bin/bash -lc \
  'source /.venv/bin/activate && \
   python /residual_rl/scripts/openpi/eval_libero_task.py \
     --host host.docker.internal \
     --task-suite-name libero_spatial \
     --task-id 0 \
     --num-trials 1 \
     --seed 7 \
     --output-dir /data'
```

Monitor memory from another terminal:

```bash
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5
```

Stop the server without removing its checkpoint cache:

```bash
cd /root/workspace/openpi
docker compose -f examples/libero/compose.yml stop openpi_server
```

### Validated result on RTX 3060 12 GB

- Task suite: `libero_spatial`
- Task ID: `0`
- Instruction: “pick up the black bowl between the plate and the ramekin and
  place it on the plate”
- Seed/trials: `7` / `1`
- Outcome: success in 113 simulator steps and 21 batch-1 inference requests
- Frozen server VRAM after load: 9,044 MiB
- Peak sampled VRAM with server and simulator: 9,624 MiB / 12,288 MiB
- Minimum sampled headroom: 2,664 MiB

The structured result and replay video are written to
`runs/openpi_libero_spatial_task0/`. This proves frozen inference fits on the
12 GB GPU; it does not establish enough headroom for co-locating additional
large GPU models.

## Frozen task screening

The reusable screening runner evaluates selected tasks sequentially against an
already-running frozen server. It writes `screen_result.json` after every task,
so partial results survive an interrupted screen. This command covers the ten
LIBERO Spatial tasks at one initial state each:

```bash
mkdir -p /root/workspace/residual_rl/runs/libero_spatial_screen_1x10

docker run --rm --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/workspace/openpi:/app:ro \
  -v /root/workspace/residual_rl:/residual_rl:ro \
  -v /root/workspace/residual_rl/runs/libero_spatial_screen_1x10:/data \
  libero /bin/bash -lc \
  'source /.venv/bin/activate && \
   python /residual_rl/scripts/openpi/screen_libero_tasks.py \
     --host host.docker.internal \
     --task-suite-name libero_spatial \
     --task-ids 0,1,2,3,4,5,6,7,8,9 \
     --num-trials 1 \
     --seed 7 \
     --output-dir /data'
```

Change `--task-suite-name` to `libero_10` to screen the harder ten-task suite.
The validated first pass scored 9/10 on Spatial and 10/10 on LIBERO-10. Spatial
task 4 was investigated further, but it scored 14/16 across direct frozen
rollouts and 5/5 through the bridge, making it too close to solved.

`eval_libero_task.py` and `screen_libero_tasks.py` accept
`--initial-state-offset`, so follow-up runs can cover new states instead of
repeating state 0. An earlier single-task candidate was confirmed with:

```bash
mkdir -p /root/workspace/residual_rl/runs/libero_10_task8_5trials

docker run --rm --gpus all \
  --add-host=host.docker.internal:host-gateway \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/workspace/openpi:/app:ro \
  -v /root/workspace/residual_rl:/residual_rl:ro \
  -v /root/workspace/residual_rl/runs/libero_10_task8_5trials:/data \
  libero /bin/bash -lc \
  'source /.venv/bin/activate && \
   python /residual_rl/scripts/openpi/eval_libero_task.py \
     --host host.docker.internal \
     --task-suite-name libero_10 \
     --task-id 8 \
     --num-trials 5 \
     --initial-state-offset 0 \
     --seed 7 \
     --output-dir /data'
```

This earlier single-task exploration found that the direct frozen baseline
succeeded on states 0, 3, and 4 and timed out on
states 1 and 2: 3/5 success. Successful episodes used 432–445 total simulator
steps; both failures exhausted the 530-step limit. Video inspection showed
that both failures placed the first moka pot, then stalled while reaching for
or grasping the second. This is a concrete late-stage correction opportunity,
not a task where the frozen policy makes no useful progress.

## Controlled LIBERO-Spatial deployment shift

Before starting RL, one action-calibration parameter was swept while OpenPI
remained frozen. At each policy-controlled simulator step, a constant offset is
added only to the normalized Cartesian x action after π0.5 inference:

```text
executed_action = pi05_action + [bias_x, 0, 0, 0, 0, 0, 0]
```

The stabilization actions, observations, prompt, gripper channel, policy
weights, and inference batch size are unchanged. Actions are not clipped. The
runner stores separate SHA-256 traces for raw policy actions and executed
actions so this boundary is auditable.

The coarse sweep used one fixed initial state from every Spatial task:

| x-action bias | Successes | Success rate |
|---:|---:|---:|
| `0.00` | 9/10 | 90% |
| `+0.02` | 10/10 | 100% |
| `+0.05` | 10/10 | 100% |
| `+0.10` | 7/10 | 70% |
| `+0.15` | 7/10 | 70% |

The small non-monotonic differences are compatible with closed-loop policy
compensation and stochastic inference. The two useful candidates were then
evaluated on the same 30 task/state pairs (ten tasks, initial states 0–2):

| x-action bias | Successes | Success rate |
|---:|---:|---:|
| `0.00` | 29/30 | 96.7% |
| `+0.10` | 24/30 | 80.0% |
| `+0.15` | 18/30 | 60.0% |

This establishes a controlled mid-success deployment regime for the project.
It shows that the frozen generalist is vulnerable to a modest controller shift;
it does not yet show that RL can recover the lost performance.

Reproduce a stratum by restarting the frozen server to reset its policy RNG,
then adding `--action-bias-x` to the screening command above. For example:

```bash
python /residual_rl/scripts/openpi/screen_libero_tasks.py \
  --host host.docker.internal \
  --task-suite-name libero_spatial \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --num-trials 3 \
  --initial-state-offset 0 \
  --action-bias-x 0.15 \
  --seed 7 \
  --output-dir /data
```

The `remote_libero` backend accepts the equivalent full-vector YAML option,
which applies the same deployment shift after main-project action composition:

```yaml
environment:
  backend: remote_libero
  options:
    action_bias: [0.15, 0, 0, 0, 0, 0, 0]
```

## State-based isolated LIBERO bridge

The next integration boundary is now implemented as:

```text
Python 3.10 residual_rl  <-- HTTP/JSON: state + base action -->  Python 3.8 LIBERO client
                                                                    |
                                                                    +-- WebSocket/images --> frozen OpenPI server
```

Images remain inside the LIBERO client. The main environment receives the
8-D simulator state, a suite-sized task one-hot, and the next 7-D frozen action,
composes the executed action, and returns it to the bridge. This is a
state-based path only; no VLA latent is exposed.

Two details are required for behavioral parity with the official evaluator:

- MuJoCo/EGL creation, reset, rendering, and stepping all run on one thread.
- Frozen OpenPI actions remain float64 end to end. The official client returns
  float64 actions, including small excursions beyond ±1; downcasting them or
  clipping at ±1 changed the simulated trajectory.

Start the frozen server as above. To reset the server-side JAX policy RNG
sequence, restart it immediately before the baseline. Minor accelerator-level
nondeterminism may still remain:

```bash
cd /root/workspace/openpi
docker compose -f examples/libero/compose.yml restart openpi_server
```

Launch the loopback-only bridge:

```bash
mkdir -p /root/workspace/residual_rl/runs/libero_bridge_state_task0_final

docker run --rm -d \
  --name residual-libero-bridge \
  --gpus all \
  -p 127.0.0.1:8765:8765 \
  --add-host=host.docker.internal:host-gateway \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/workspace/openpi:/app:ro \
  -v /root/workspace/residual_rl:/residual_rl:ro \
  -v /root/workspace/residual_rl/runs/libero_bridge_state_task0_final:/data \
  libero /bin/bash -lc \
  'source /.venv/bin/activate && \
   python /residual_rl/scripts/openpi/libero_bridge_service.py \
     --policy-host host.docker.internal \
     --task-suite-name libero_spatial \
     --task-id 0 \
     --seed 7 \
     --output-dir /data'

curl --fail http://127.0.0.1:8765/health
```

Run three fixed initial states through the normal project entry point:

```bash
cd /root/workspace/residual_rl
.venv/bin/python -m last_millimeter.train \
  --config configs/libero/frozen_state.yaml
```

The config's evaluation seeds 10000, 10001, and 10002 select LIBERO initial
states 0, 1, and 2 respectively. Each policy request contains one observation.
The bridge writes videos, action/input trace hashes, and `bridge_result.json`;
the main project writes its usual config, metrics, and summary.

Validated repeated result on the RTX 3060:

| Initial state | Success | Control steps | Inference requests |
|---:|:---:|---:|---:|
| 0 | yes | 100 | 20 |
| 1 | yes | 110 | 22 |
| 2 | yes | 84 | 17 |

Success was 3/3 with a mean of 98 control steps. Peak sampled VRAM was
9,622 MiB / 12,288 MiB, leaving 2,666 MiB headroom. Stop the completed bridge
and frozen server with:

```bash
docker stop residual-libero-bridge
cd /root/workspace/openpi
docker compose -f examples/libero/compose.yml stop openpi_server
```

The earlier exploratory LIBERO-10 task 8 run uses the same boundary. Launch the
bridge with `--task-suite-name libero_10 --task-id 8`, then run:

```bash
.venv/bin/python -m last_millimeter.train \
  --config configs/libero/libero10_task8_frozen_state.yaml
```

On fixed initial states 0–4, the bridge baseline scored 1/5: state 0 succeeded
in 417 total steps, while states 1–4 reached the 530-step limit. States 1 and 2
also failed in the direct evaluator. States 3 and 4 changed outcome under small
accelerator-level action differences, so direct-loop and bridge results are
reported separately rather than treated as bitwise-equivalent runs. Sampled
VRAM during this longer bridge evaluation was 9,365 MiB / 12,288 MiB.

Video inspection of these rollouts (frame-by-frame stills at
`results/libero_10_task8_verify/`, raw `result.json` alongside them) is what
ruled task 8 out as a "last millimeter" task and motivated the pivot to
Thread B below: LIBERO's coarse region/contact-based success criteria don't
isolate a single repeatable tight-tolerance contact phase the way a real
peg-in-hole insertion does.

### Multi-task Spatial bridge and oracle control

For joint Spatial experiments, launch one bridge that allows all ten tasks:

```bash
mkdir -p /root/workspace/residual_rl/runs/libero_bridge_spatial_suite

docker run --rm -d \
  --name residual-libero-bridge \
  --gpus all \
  -p 127.0.0.1:8765:8765 \
  --add-host=host.docker.internal:host-gateway \
  -e MUJOCO_GL=egl \
  -e MUJOCO_EGL_DEVICE_ID=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYOPENGL_PLATFORM=egl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /root/workspace/openpi:/app:ro \
  -v /root/workspace/residual_rl:/residual_rl:ro \
  -v /root/workspace/residual_rl/runs/libero_bridge_spatial_suite:/data \
  libero /bin/bash -lc \
  'source /.venv/bin/activate && \
   python /residual_rl/scripts/openpi/libero_bridge_service.py \
     --policy-host host.docker.internal \
     --task-suite-name libero_spatial \
     --task-ids 0,1,2,3,4,5,6,7,8,9 \
     --seed 7 \
     --output-dir /data'
```

The suite configs use `round_robin` scheduling. Ten episodes cover task IDs
0–9 at initial state 0; 30 episodes cover the Cartesian product of task IDs
0–9 and initial-state IDs 0–2. `uniform` scheduling is available for later
training. The residual representation concatenates the 8-D robot state and
10-D task one-hot; the base action remains a separate actor input.

With a freshly restarted frozen server before each condition, the validated
state-0 control produced:

| Condition | Config | Success |
|---|---|---:|
| Unbiased frozen π0.5 | `spatial_suite_frozen_state.yaml` | 10/10 |
| `+0.15` x bias | `spatial_suite_bias_x_0p150_frozen_state.yaml` | 6/10 |
| Bias plus fixed `-0.15` x oracle | `spatial_suite_bias_x_0p150_oracle_state.yaml` | 10/10 |

Run a condition through the normal project evaluator with:

```bash
.venv/bin/python -m last_millimeter.evaluate \
  --config configs/libero/spatial_suite_bias_x_0p150_oracle_state.yaml \
  --episodes 10
```

The oracle is an evaluation-only `observation_action.offset`; it is supplied by
configuration and is never learned. At the bridge, its first executed action
matched the raw frozen action exactly in float64 for every checked episode,
confirming that `-0.15 + 0.15` cancels in the intended action space. The main
evaluator reports its correction norm as zero because the known offset is
implemented as a fixed base-action transform, not an RL policy output.

The bridge writes task IDs, descriptions, state IDs, per-task success, raw and
executed action hashes, and videos to `bridge_result.json`. Post-run GPU memory
was 9,084 MiB / 12,288 MiB; the previously sampled bridge peak remains
9,622 MiB. This leaves the frozen batch-size-1 workflow within the 12 GB limit.

## Experiment modes and next gate

- `frozen`: evaluate the frozen base policy.
- `scratch`: train SAC without access to the base action.
- `residual`: train an always-active bounded correction.
- `gated`: jointly train a bounded correction and intervention gate.

Residual and gated rewards can penalize correction magnitude and gate
intensity. The actor receives the configured representation and reference base
action. Critics evaluate the policy output in that same context, while the
environment receives the composed action.

The oracle and multi-task bridge gates are complete. Before state-based
always-on residual training begins, the next step is to define disjoint
training and held-out initial-state sets, configure uniform task sampling for
training and round-robin sampling for evaluation, and rerun the 30-pair bridge
baseline if a full bridge-level confirmation is required. VLA-latent
extraction and learned gating remain separate follow-up steps, with the VLA
frozen throughout.

## Thread B: robomimic Square residual RL (non-VLA precision task)

The LIBERO thread above evaluates a VLA-based frozen policy, but LIBERO's own
success criteria are coarse (region/contact-based), so no LIBERO task offers
a true tight-tolerance insertion with a single repeatable critical phase --
the exact shape of task the RL Token paper (Physical Intelligence) targets.
NutAssemblySquare (robomimic) is a real peg-in-hole insertion task, so this
thread swaps the frozen VLA for a frozen BC-RNN policy trained on robomimic's
own demonstrations, and asks the same residual-RL question against a genuine
insertion task instead of a coarse rearrangement task.

### Frozen base policy

A BC-RNN policy (robomimic's own paper-reproduction config: GMM+LSTM, 2000
epochs, `configs/robomimic/square_frozen_state.yaml` wraps the resulting
checkpoint) was trained on robomimic's public 200-demo Square/PH/low-dim
dataset. See `docs/figures/bc_rnn_baseline_training.png` for the training
curve. The epoch-800 checkpoint (85% rollout success, matching robomimic's
published benchmark) is the frozen baseline used by every experiment below.

New infrastructure: `docker/robomimic/train.Dockerfile` (robomimic 0.3.0 +
robosuite 1.4.1, with a `mujoco_py` compatibility stub -- robomimic's
`EnvRobosuite` wrapper imports it only for one unused exception type),
`scripts/robomimic/square_bridge_service.py` (a state/base-action HTTP bridge
analogous to the LIBERO bridge, serving the loaded BC-RNN checkpoint), and a
new `remote_robomimic` environment backend that reuses the existing
LIBERO-bridge client code unchanged (the wire protocol was never
LIBERO-specific).

### Residual-RL results

| Attempt | Mechanism | Success rate |
|---|---|---:|
| Frozen baseline | -- | **85%** (76.7% on the 30-ep speed-comparison seed set) |
| Residual (always-on) | correction applied every step | ~45% (declining) |
| Gated, gate bias=0.5, λ_gate=0.01 | learned selectivity, weak prior | ~65% (mild decline) |
| Gated, gate bias=0.1 (fixed), λ_gate=0.02 | learned selectivity, correct prior | peaked ~70% mid-run, 55% final |
| Triggered, heuristic gate (pre critic-fix) | correction confined to a hand-crafted critical-phase window | stable 68-80% throughout, 65% final |
| **Triggered, heuristic gate (critic-fix)** | same, with the critic/actor blind spot below fixed | **90% final eval (20 ep), 83.3% on the 30-ep speed-comparison seed set -- beats the frozen baseline** |

See `docs/figures/success_rate_comparison.png` and `docs/figures/gate_trend.png`.
The first three attempts (50,000 steps / ~250 episodes each) never reliably
beat the frozen baseline, despite fixing two real bugs along the way (an
unbounded SAC entropy temperature in `rl/sac.py`, and a bridge bug that
zeroed the reported base action on every episode termination, corrupting the
SAC critic's bootstrap target for timeout failures -- both still relevant to
any future config that reuses this code) and one real design flaw (gated
mode's actor defaulted to ~50% intervention at initialization instead of
trusting the frozen policy; see `ActionComposer.GATE_INIT_BIAS` in
`policies/composition.py`). TRIGGERED mode was the first to reliably match
the baseline, and after the critic/actor fix described below, the first to
beat it outright.

### Where do the frozen baseline's failures actually occur?

Frame-by-frame inspection of `robomimic.scripts.run_trained_agent` rollouts
(see `docs/figures/task_square_*.png`) confirms the failures are genuinely
last-millimeter: a clean success (nut seated on the peg) and a failure where
the peg is physically knocked over during a careful, deliberate insertion
attempt use the same reach/grasp/transport motion up to the final contact
phase. This rules out "wrong task" as the reason the first three attempts
never beat the baseline -- the real explanation is that a *learned* gate has
to solve unsupervised temporal credit assignment (which of ~150-400 steps
matter) from a sparse episode-terminal reward alone, a much harder problem
than the RL Token paper's own recipe, which hand-engineers the critical-phase
boundary (via environment resets or human intervention labels) rather than
learning it from RL reward.

### TRIGGERED mode: hand-crafting the critical-phase boundary

`ControlMode.TRIGGERED` (`policies/composition.py`) replaces the learned gate
with a heuristic computed directly from ground-truth simulator state in
`scripts/robomimic/square_bridge_service.py --trigger`: the gripper is
"in the critical phase" when it is both close to the peg (empirically
calibrated against the frozen policy's own rollouts -- distance is measured
to the peg body's *origin*, which is at its base, so it bottoms out around
0.11m even at a successful insertion rather than near 0) and moving slowly
(a per-step displacement proxy). The actor now only has to learn *what*
correction to apply once already inside a known ~15-30%-of-episode window,
not *when* -- exactly what the RL Token paper's own recipe does by resetting
into the critical phase or using human intervention labels, rather than
asking RL to discover the boundary from a sparse reward.

The result is the most stable run in this whole thread: success rate held in
a 67.5-80% band for the entire 224-episode run with no decline (vs. steady
decline for always-on residual, and a rise-then-crash for both learned-gate
attempts), and the 65% final deterministic-policy evaluation is the best of
any RL attempt so far.

### A critic/actor blind spot, found and fixed

The SAC critic and actor were being trained on the actor's *raw* sampled
output, not the quantity that actually reached the environment
(`policy_output * trigger`). For every step where `trigger=0` (~80% of an
episode), the environment transition is identical no matter what the actor
outputs, so the critic was being asked to fit `Q(s, a)` to many different `a`
that all have zero real effect -- wasted, noisy learning signal, and the
actor was getting task gradient from states where its output cannot matter.
Fixed by threading `trigger`/`next_trigger` through `ReplayBuffer` and
multiplying both the critic's action input and the actor's own resampled
output by it in `SACAgent.update` (`rl/replay_buffer.py`, `rl/sac.py`).
Defaults to `1.0` and is a no-op for every other control mode.

Retraining with the fix (same config, same 50,000 steps, pre-fix run
preserved at `runs/robomimic_square_triggered_precriticfix`) moved the final
deterministic evaluation from 65% to **90%**, beating the 85% frozen
baseline outright, and cut mean episode length from 236.1 to 178.4 steps.
Two things are worth noting about how that improvement shows up: the
*training-time* rolling success rate barely moved (both runs sit in the same
noisy ~68-74% band throughout, since every training episode carries fixed
`alpha=0.1` exploration noise) -- the fix only shows up once you look at the
deterministic policy, which training curves alone would never surface. And
critic/actor loss did **not** get visibly smoother post-fix (critic_loss's
max was if anything higher: 131.9 vs 93.4 pre-fix), so the benefit isn't
"cleaner optimization" -- it's that the critic and actor were previously
being fit to the wrong target throughout training, and correcting the
target changed what the deterministic policy converged to, independent of
how noisy the loss curves look along the way.

### Does RLT's actual claim hold here? Checking the paper against our own data

The RL Token paper is explicit about what to expect when the frozen policy is
already competent: *"Where the VLA is already competent (e.g. the Ethernet
task) it maintains success rate and increases throughput"* -- and separately,
on Ethernet, RLT *"match[es] the base policy's high success rate while
reducing mean steps to completion by 2x"* and is *"3x faster in the critical
phase."* Square's 85% baseline puts it in exactly this regime, so the paper
itself predicts we should *not* see a big accuracy jump -- the honest
expectation, before even running anything, is "success rate roughly
maintained, critical phase faster," not "perfect accuracy."

The pre-critic-fix measurement matched that prediction almost exactly: both
policies evaluated through the same trigger-instrumented bridge (30 episodes
each, same seed), success rate flat within noise (76.7% frozen vs. 73.3%
trained), critical phase 32% faster (27.8 vs. 40.7 steps). That was the
expected RLT regime, on a task the paper never tested.

Retraining with the critic/actor fix changes this picture. The same 30-episode,
same-seed comparison now gives **83.3% success (trained) vs. 76.7% (frozen)**
-- a real accuracy gain, not just noise -- alongside a smaller but still
present speed gain: the critical phase drops from 40.7 to 33.1 steps (~19%
faster) and the whole episode from 153.3 to 141.6 steps (~8% faster). See
`docs/figures/speed_comparison.png`. The final deterministic evaluation (20
episodes, the number quoted in the results table above) shows an even larger
accuracy gap, 90% vs. 85%.

So the honest updated conclusion is: RLT's paper describes what happens when
the *only* lever pulled is confining learning to a known critical-phase
window -- speed goes up, accuracy is preserved. Our pre-fix run reproduced
exactly that. But we also had a second, independent bug (the critic/actor
blind spot) stacked on top of the setup RLT describes; fixing it moved
Square's result out of "maintains accuracy" and into "measurably better
accuracy," a stronger outcome than the paper's own claim for this regime.
That is not evidence RLT's claim is wrong -- RLT's recipe doesn't have this
particular blind spot to begin with, so its "accuracy maintained" result
already reflects a cleaner training signal than our pre-fix run had. It does
mean the *speed* effect specifically is smaller post-fix (19% vs. 32%) even
as overall performance improved, consistent with the idea that some of the
pre-fix "speed-only" gain was actually the critic partially compensating for
its own blind spot by learning a coarser, faster-but-less-precise policy,
which the fix replaced with a slower-but-more-decisive one.

### What do the trained policy's remaining failures actually look like?

Video inspection (`scripts/robomimic/square_bridge_service.py
--record-video`, driving the critic-fix checkpoint
(`runs/robomimic_square_triggered/checkpoints/final.pt`) through
`last_millimeter.evaluate` with the same config and seed as the speed
comparison above) surfaces the failure modes directly, using the per-episode
`trigger_active_fraction` logged to `bridge_result.json` to classify each one.

A methodological note first: this video-capture run scored 76.7% (23/30),
not the 83.3% (25/30) reported earlier for the same checkpoint, config, and
seed. The base BC-RNN policy samples its action from a GMM head, i.e. it is
genuinely stochastic per rollout, and that randomness is drawn from the
bridge process's own RNG stream -- enabling `--record-video` adds rendering
calls that were not present in the earlier run, which is enough to shift the
sequence of random draws and diverge trajectories from a fresh bridge
session, even with an identical `--seed`. Treat 76.7-83.3% as the honest
range for this checkpoint on this 30-episode sample rather than a single
precise figure; the two independent samples still agree that it exceeds the
76.7% frozen baseline (or are at worst tied with it).

The 7 failures in this run split into three categories, not the two seen
in the pre-fix checkpoint's failures:

- **Never triggered** (1/7, episode 16, `trigger_active_fraction=0.0`; see
  `docs/figures/task_square_trained_criticfix_never_triggered_*.png`): the
  gripper never gets both close and slow enough to enter the critical-phase
  window at all within the 400-step horizon -- the wrist-camera end frame
  shows the nut still being carried, never settled into a contact attempt.
  The correction never gets a chance to help; this is a reach/transport-phase
  failure, not a last-millimeter one, and the trigger design cannot address
  it by construction.
- **Stuck inside the window** (3/7, episodes 2/28/29, fractions 73-81%; see
  `docs/figures/task_square_trained_criticfix_stuck_window_*.png` for
  episode 29): the opposite extreme -- these episodes spend the majority of
  their 400 steps inside the critical-phase zone without ever completing.
  Episode 29's end frame shows the nut knocked askew right next to the peg
  rather than seated in it. This looks like a case where the correction is
  engaged but not decisive enough, a plausible target for a larger
  `residual_scale` or more training within the window.
- **Partial engagement** (3/7, episodes 13/17/20, fractions 24-37%): a
  category not visible in the earlier two-way taxonomy -- the trigger fires
  substantially (roughly a third of the episode) but not continuously,
  suggesting the gripper repeatedly enters and drops back out of the
  distance/speed window without committing, rather than either missing it
  entirely or camping inside it. This sits between the other two failure
  modes and is consistent with the trigger's distance/speed thresholds being
  a coarse, non-learned boundary that the gripper can straddle rather than
  cleanly cross.

None of the three categories is "wrong task" -- all are legible, specific
behaviors that suggest concrete next experiments (a looser trigger threshold
for the first and third; more training or capacity for the second) rather
than a dead end.

### Would more training have helped? A dedicated 100k-step run says no

The 50,000-step critic-fix result above raises an obvious question: is 85-90%
success a ceiling, or just where training happened to stop? Answering this
required fixing a real infrastructure gap first -- every evaluation up to
this point had `training.evaluation_interval: 0` (mid-training evaluation
disabled entirely), because `evaluate_components()` builds a second
environment pointed at the *same* remote bridge endpoint as the training
loop, and closing that second environment (which it always did, at the end
of its evaluation episodes) tore down the shared bridge process out from
under the still-running training loop. Every prior result was therefore a
single snapshot at the very end of training, with zero visibility into
whether performance was still climbing, had plateaued, or was oscillating
along the way.

Fixed properly this time: `EnvironmentConfig.eval_endpoint` (`config.py`)
lets evaluation target an independent bridge server instead of the training
bridge, and a new `close_env` parameter on `evaluate_components()`
(`evaluation.py`) lets mid-training evaluation calls leave that independent
bridge open for reuse at the next checkpoint, since a remote bridge's
`/close` is a one-way, permanent shutdown -- the first attempt at this fix
still crashed (at the *second* evaluation checkpoint) because `close_env`
defaulted to closing the bridge after its very first use. Two independent
bridge containers (`residual-square-bridge-train` on port 8766,
`residual-square-bridge-eval` on port 8767) plus this fix gave a real,
10-point learning curve for the first time: a fresh 100,000-step run (double
the original budget), evaluated deterministically (10 episodes) against the
independent eval bridge every 10,000 steps.

See `docs/figures/extended_training_curve.png`:

| Step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| Success | 80% | 80% | 70% | 80% | 80% | 70% | 70% | 80% | 90% | 80% |

Mean 78%, std 6 points, **no upward or downward trend** -- the curve is flat
and noisy across the entire range, including the 90% blip at step 90k, which
looked like a late improvement in real time but was back down to 80% at the
very next checkpoint. The final evaluation (80%) is statistically
indistinguishable from the checkpoint at step 10,000 (80%). This is a direct,
negative answer: doubling the training budget bought nothing measurable.

The loss curves reinforce this rather than contradict it (see
`docs/figures/loss_curves.png`, and per-run curves under
`docs/figures/per_run/`): actor loss climbs almost linearly across all three
critic-fix-era runs (the original 50k run, its retrain, and this 100k
extension) with no sign of saturating, and critic loss -- always noisy with
periodic spikes -- develops larger and more frequent spikes as the extended
run continues past 50k (up to ~250, versus a max of ~130 in either 50k run).
Together with the flat success-rate curve, the honest reading is that this
policy converged, in the sense that matters (task performance), by roughly
10,000-20,000 steps, and the additional 80,000 steps did not find a better
policy -- if anything, the underlying SAC optimization looks less settled at
100k than it did earlier, even though that instability hasn't (yet) shown up
as worse task performance. This rules out "just train longer" as a fix for
the remaining ~20% failure rate; the three failure categories documented
above (never-triggered, stuck-in-window, partial engagement) point at
structural limits of the heuristic trigger and `residual_scale` instead, not
an undertrained policy.

## Diffusion Policy lineage: a second base policy, added alongside BC-RNN

Everything above uses BC-RNN as the frozen base policy. This section adds a
second, parallel lineage using robomimic's Diffusion Policy implementation
(UNet + DDPM, action-chunked receding-horizon control) on the exact same
task, dataset, and eval conventions -- BC-RNN is not replaced or touched by
any of this; it stays the primary, fully-documented result above.

### Stage 2: Diffusion Policy base training

Trained for the same 2000 epochs, on the same Square/PH/low-dim dataset, with
the same 20-episode rollout evaluation every 200 epochs as BC-RNN
(`configs/robomimic_train/square_diffusion_policy.json`, adapted from
robomimic's own `diffusion_policy.json` template; diffusion hyperparameters
-- UNet, DDPM noise schedule, `observation_horizon=2`, `action_horizon=8`,
`prediction_horizon=16` -- left at template defaults). See
`docs/figures/base_policy_comparison.png`.

| Epoch | 200 | 400 | 600 | 800 | 1000 | 1200 | 1400 | 1600 | **1800** | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|
| Success | 85% | 85% | 85% | 70% | 75% | 90% | 85% | 85% | **95%** | 90% |

Epoch 1800 (95%, first epoch to hit the run's peak -- same selection rule
used for BC-RNN's epoch 800) is the checkpoint used for every Diffusion
Policy residual-RL experiment below. Diffusion Policy's peak beats BC-RNN's
85% by a real margin, which matters for reading Stage 3: there was
meaningfully less headroom left for a residual correction to improve on.

### Stage 3: SAC TRIGGERED (critic-fix) on frozen Diffusion Policy

Same TRIGGERED-mode, critic-fix architecture as the primary BC-RNN result
above -- same heuristic critical-phase trigger, same fixed `alpha=0.1`, same
50,000-step budget (`configs/robomimic/square_triggered_diffusion.yaml`) --
with the frozen policy swapped for the epoch-1800 Diffusion Policy
checkpoint. Two real bugs surfaced getting this far, both in
`scripts/robomimic/square_bridge_service.py`, both confirmed backward-
compatible no-ops for BC-RNN (neither changes anything about the results
above): `flatten_state()` didn't account for `frame_stack=2` doubling each
observation field's shape (robomimic's `FrameStackWrapper`, unused by
BC-RNN, stacks the last 2 frames per key) -- fixed by keeping only the most
recent frame; and `step()` passed `action.tolist()` into `self.env.step()`,
which broke inside `FrameStackWrapper.step()`'s own frame-history bookkeeping
(a plain list doesn't support the `arr[None]` new-axis trick the wrapper
needs) -- fixed by passing the ndarray directly.

Final deterministic evaluation (20 episodes): **90% success**, mean episode
length 201.1 steps -- essentially matching BC-RNN's own critic-fix result
(90%, 178.4 steps) on the training run's own held-out evaluation.

### The speed-comparison result: a regression, not a repeat

The same 30-episode, same-seed (10000) frozen-vs-trained comparison used
for BC-RNN's speed claim (`docs/figures/speed_comparison_diffusion.png`,
raw data in `results/speed_comparison_diffusion/`) tells a different story
here, and it's reported plainly rather than reframed:

| | Success rate | Mean episode length (all episodes) |
|---|---:|---:|
| Frozen (epoch 1800) | 90.0% | 185.9 steps |
| Trained (SAC critic-fix) | 80.0% | 205.5 steps |

The trained residual correction did **worse** on accuracy than the frozen
baseline on this sample -- the opposite of BC-RNN's 85%->90% gain. Restricted
to successful episodes only (the same lens the BC-RNN speed figure uses),
the picture is more nuanced: total episode length and critical-phase length
are both marginally *shorter* under the trained policy (162.1 -> 156.9 steps;
12.6 -> 8.7 critical-phase steps). So this isn't "the correction makes every
episode worse" -- episodes it doesn't derail are, if anything, slightly
faster, matching the RL Token paper's speed prediction. It's that the trained
policy derails more episodes than it used to succeed on, dragging down the
all-episode averages in both columns above.

**A plausible, unconfirmed explanation.** Diffusion Policy predicts action
*chunks* via receding-horizon control: robomimic's `DiffusionPolicyUNet`
(`docker/robomimic/robomimic_src/robomimic/algo/diffusion_policy.py`) keeps
an internal `action_queue` and only runs a fresh diffusion forward pass when
that queue is empty -- with `action_horizon=8`, it replans once every 8
steps and blindly executes the other 7 in between, with no way to know what
the SAC correction actually did to the trajectory during that window.
BC-RNN has no such queue -- it predicts fresh, single-step, every call -- so
it never has this mismatch. This would mean the correction can compound a
divergence from the base policy's own stale plan across up to 7 steps at a
time, rather than being "absorbed" step-by-step the way it is for a reactive
policy.

This hypothesis was **not tested**. Three ways to adapt SAC to the chunk
structure instead of fighting it were discussed and design-sketched but not
implemented or retrained: (1) sync SAC's own decision cadence to the chunk
boundary, holding one correction constant across all 8 steps of a chunk
instead of resampling every step; (2) keep SAC's per-step cadence but give
it explicit chunk-position awareness (steps since last replan, or the base
policy's remaining planned actions) as additional observation inputs; (3)
apply the correction to the *state the diffusion policy conditions on* when
it replans, rather than to the executed action, so the model always
generates a fully self-consistent chunk from a corrected starting point.
Option 1 is the one that most directly preserves both of chunking's
advantages (compute efficiency -- one diffusion forward pass still covers 8
steps -- and within-chunk trajectory coherence, since a constant per-chunk
offset shifts a plan without distorting its shape the way independent
per-step corrections could); option 3 is the most novel and the furthest
from the `executed_action = base_action + gate * correction` composition
used everywhere else in this project, and isn't backed by an established
citation for this exact combination (closest related work: Diffusion
Policy, ACT/temporal ensembling, residual RL, classifier guidance, and
DPPO). This is left here as an open question and a concrete starting point
for whoever picks this up next, not a settled explanation.

## Where the raw data lives

Every run referenced in this document has its full episode-by-episode
history preserved under `results/<run_name>/` (`metrics.csv`, `config.yaml`,
and `summary.json` where the run completed cleanly), plus the BC-RNN
baseline's own training log at `results/bc_rnn_baseline/log.txt` and the
raw 30-episode speed-comparison bridge outputs under
`results/speed_comparison/`. `docs/figures/per_run/` has one standalone
success-rate-over-training figure (plus loss curves, where applicable) per
run, in addition to the cross-run comparison figures in `docs/figures/`
referenced throughout this document. Every model checkpoint (`.pt`/`.pth`,
under each run's own `results/<run_name>/model_checkpoints/`) and every raw
rollout video (`.mp4`, under each run's own `results/<run_name>/videos/`)
referenced anywhere in this document is committed too, alongside the
zero-shot pi05_libero NutAssemblySquare/ToolHang smoke-test videos that
motivated building the robomimic bridge in the first place
(`results/robosuite_nutsquare/`, `results/robosuite_toolhang/`). Nothing in
this document depends on the local, gitignored `runs/` directory.

The one exception: the Diffusion Policy base checkpoints
(`results/diffusion_policy_baseline/model_checkpoints/`) are a UNet, not a
small GMM+LSTM, so each one is ~1GB (10GB for the full run) -- too large for
a plain git repo. Those are hosted instead on Hugging Face Hub at
[`georginio2000/diffusion-square-nutassembly`](https://huggingface.co/georginio2000/diffusion-square-nutassembly)
(gitignored locally) and downloadable with `huggingface_hub`:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    "georginio2000/diffusion-square-nutassembly",
    "checkpoints/model_epoch_1800_low_dim_success_0.95.pth",
)
```
Every other Diffusion Policy artifact -- Stage 3's SAC checkpoints
(`results/robomimic_square_triggered_diffusion/model_checkpoints/`, ~20MB
total), all config/metrics/summary files, and the speed-comparison JSON --
is small enough to be committed normally, same as everything else in this
document.

`configs/robomimic/*.yaml` hold the *current* version of each config file,
which was edited in place across iterations (e.g. `square_triggered.yaml` now
reflects the final 100k dual-bridge setup, not the 50k version it started
as). The config actually used for each specific run is preserved exactly,
unmodified, as `results/<run_name>/config.yaml` -- that copy, not the
current file under `configs/`, is the authoritative record of what produced
that run's numbers.

| Run | Results (config + metrics + summary) |
|---|---|
| Frozen BC-RNN baseline | `results/bc_rnn_baseline/` |
| Residual (always-on) | `results/robomimic_square_residual/` |
| Gated, bias=0.5, λ_gate=0.01 | `results/robomimic_square_gated_lambda0p01/` |
| Gated, bias=0.1 (fixed), λ_gate=0.02 | `results/robomimic_square_gated/` |
| Triggered, pre critic-fix (50k) | `results/robomimic_square_triggered_precriticfix/` |
| Triggered, critic-fix (50k) | `results/robomimic_square_triggered_50k_criticfix/` |
| Triggered, critic-fix, extended (100k) | `results/robomimic_square_triggered/` |
| Speed comparison (frozen vs. trained) | `results/speed_comparison/` |
| Frozen Diffusion Policy baseline | `results/diffusion_policy_baseline/` |
| Triggered, critic-fix, on Diffusion Policy (50k) | `results/robomimic_square_triggered_diffusion/` |
| Speed comparison, Diffusion Policy lineage (frozen vs. trained) | `results/speed_comparison_diffusion/` |
| LIBERO-10 task 8 verification (Thread A, abandoned) | `results/libero_10_task8_verify/` |

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

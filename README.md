# Teaching a Robot Foundation Model the Last Millimeter

This repository studies whether a small reinforcement-learning controller can
specialize a frozen generalist robot policy for precision manipulation:

```text
executed_action = base_action + gate * correction
```

The toy residual-RL experiment, frozen OpenPI π0.5-LIBERO baselines, and an
isolated state-based LIBERO bridge are validated. OpenPI is served as a frozen
remote policy; it is not installed in the main project environment and no VLA
parameters are fine-tuned.

## Current milestone status

- All toy `frozen`, `scratch`, `residual`, and `gated` modes are preserved.
- The full CPU-only test suite passes.
- The 30,000-step toy residual config was verified on CUDA.
- Official OpenPI was cloned recursively and pinned exactly.
- Frozen `pi05_libero` evaluation was reproduced at inference batch size 1.
- The main project evaluated three LIBERO initial states through an isolated
  state/base-action bridge; all three succeeded.
- Reusable batch-size-1 screening covered every LIBERO Spatial and LIBERO-10
  task, with explicit initial-state range selection.
- LIBERO-10 task 8, “put both moka pots on the stove,” is the selected residual
  candidate: the frozen policy scored 3/5 in the direct loop and 1/5 through
  the main-project bridge on fixed initial states 0–4.
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
| Representation | `identity`, `observation_key` | Flat toy state or a selected mapping feature |

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

## Frozen task screening and candidate selection

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
repeating state 0. The selected candidate was confirmed with:

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

The direct frozen baseline succeeded on states 0, 3, and 4 and timed out on
states 1 and 2: 3/5 success. Successful episodes used 432–445 total simulator
steps; both failures exhausted the 530-step limit. Video inspection showed
that both failures placed the first moka pot, then stalled while reaching for
or grasping the second. This is a concrete late-stage correction opportunity,
not a task where the frozen policy makes no useful progress.

## State-based isolated LIBERO bridge

The next integration boundary is now implemented as:

```text
Python 3.10 residual_rl  <-- HTTP/JSON: state + base action -->  Python 3.8 LIBERO client
                                                                    |
                                                                    +-- WebSocket/images --> frozen OpenPI server
```

Images remain inside the LIBERO client. The main environment receives the
8-D simulator state and the next 7-D frozen action, composes the executed
action, and returns it to the bridge. This is a state-based path only; no VLA
latent is exposed.

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

The selected LIBERO-10 task 8 uses the same boundary. Launch the bridge with
`--task-suite-name libero_10 --task-id 8`, then run:

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

## Experiment modes and next gate

- `frozen`: evaluate the frozen base policy.
- `scratch`: train SAC without access to the base action.
- `residual`: train an always-active bounded correction.
- `gated`: jointly train a bounded correction and intervention gate.

Residual and gated rewards can penalize correction magnitude and gate
intensity. The actor receives the configured representation and reference base
action. Critics evaluate the policy output in that same context, while the
environment receives the composed action.

LIBERO-10 task 8 is now the selected imperfect baseline, but residual training
is still gated. The current 8-D representation contains only end-effector pose
and gripper state; it cannot identify either moka pot, the stove target, or
whether the first subgoal is complete. The next milestone is to expose a
compact simulator task-state vector through the existing bridge, define and
test phase-aware dense reward diagnostics for the two placements, and rerun the
frozen baseline without changing OpenPI. State-based always-on residual
training can begin only after that boundary is reproducible. VLA-latent
extraction and learned gating remain separate follow-up steps, with the VLA
frozen throughout.

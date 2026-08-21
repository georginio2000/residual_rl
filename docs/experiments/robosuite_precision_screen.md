# Frozen π0.5 Native Robosuite Precision Screen

## Question

Does frozen OpenPI π0.5-LIBERO show enough partial competence on a native
precision task that a small residual controller could plausibly specialize its
last-millimeter behavior?

This changes the experimental emphasis from manually perturbing positions or
dynamics to selecting a task whose unmodified success condition already
requires precise alignment, insertion, contact, and release. The earlier
LIBERO scene-shift results remain valid premise evidence, but they are not used
to select this precision task.

## Frozen boundary

- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO submodule: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`
- Policy: frozen; no fine-tuning or normalization changes
- Inference batch size: 1
- Client: isolated Python 3.8 / Torch 1.11 `libero` image
- Native simulator: robosuite 1.4.1 in that client image
- Robot/controller: Panda / 7-D `OSC_POSE`

This is a transfer screen, not an official OpenPI evaluation. The checkpoint
was trained for LIBERO rather than the native robosuite task distributions.

## Candidate order

1. `NutAssemblySquare`: first choice because reaching, grasping, lifting,
   transport above the square peg, and final placement can be separated.
2. `ToolHang`: secondary inspection because it combines stand assembly with a
   precise hang-and-release operation. An early failure may reflect long-horizon
   planning rather than precision control.

Fixed prompts:

- `NutAssemblySquare`: “place the square nut on the square peg”
- `ToolHang`: “assemble the tool stand and hang the tool on the stand”

Prompt overrides must be recorded as a new experimental condition.

## Protocol

### Phase 0: interface smoke test

Run one `NutAssemblySquare` episode at seed 7. Confirm before interpreting task
performance:

- the two videos show correctly oriented `agentview` and wrist inputs;
- state shape is 8 and action shape is 7;
- returned actions are finite and accepted without clipping or conversion;
- exactly one observation is sent per inference request;
- server plus renderer remain below 12,288 MiB VRAM;
- `result.json` contains upstream pins, reset seed, trace hashes, and stage data.

Monitor VRAM in a second shell:

```bash
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5
```

### Phase 1: initial NutAssemblySquare screen

Run ten episodes with base seed 7, producing placement seeds 7–16. Preserve
every video and the incrementally written `result.json`. Report:

- total success;
- episodes that ever reached, grasped, lifted, or hovered over the peg;
- maximum continuous robosuite reward component for every stage;
- representative failure categories from video inspection;
- peak sampled VRAM.

The stage flags are diagnostic metrics only. They are not fed to π0.5 and are
not an oracle action correction.

### Phase 2: ToolHang inspection

Run five seeds using the same frozen boundary. Record whether the frame was ever
assembled and whether the tool was ever hung. Inspect videos to distinguish
early semantic/planning failures from final alignment or release failures.

### Phase 3: confirmation

Expand the more suitable candidate to 30 frozen episodes only after the screen
shows useful partial competence. Freeze the task, prompt, camera convention,
horizon, and seed set before that confirmation run.

## Predeclared selection rule

Prefer `NutAssemblySquare` if π0.5 repeatedly lifts the nut and makes progress
toward the peg but does not solve at least 9/10 initial episodes. A zero-success
screen can still qualify if late-stage hover/alignment behavior is repeated;
zero lifts indicates an unsuitable visual/semantic transfer rather than a
last-millimeter problem.

Use `ToolHang` only if it repeatedly completes the stand or reaches the final
hanging interaction. Reject it as the first residual benchmark if most runs
fail before those stages, because that would confound precision correction with
multi-stage planning.

No residual training, shaped reward design, or VLA-latent extraction begins
until one candidate passes this gate.

## Results

Pending the frozen GPU smoke test. Raw artifacts will remain under ignored
`runs/` directories; a compact result summary will be committed here after the
run.

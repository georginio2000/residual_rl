# Teaching a Robot Foundation Model the Last Millimeter

This repository studies whether a small reinforcement-learning controller can
specialize a frozen generalist robot policy for precision manipulation. The
controller can either replace the base action, add an always-on correction, or
learn a gate that determines when to intervene:

```text
executed_action = base_action + gate * correction
```

The first implementation milestone is deliberately CPU-runnable. It validates
the complete training and evaluation path on a two-dimensional precision task
before simulator and foundation-model integration introduce additional sources
of failure.

## Implemented experiment modes

- `frozen`: evaluate the frozen proportional base policy.
- `scratch`: train SAC without access to the base policy.
- `residual`: train an always-active bounded action correction.
- `gated`: jointly train a bounded correction and intervention gate.

Residual and gated rewards can penalize correction magnitude and gate intensity.
The actor receives the state representation and reference base action. The
critics evaluate the policy's control output (correction and optional gate) in
that same context; the environment receives the composed action. This avoids
aliasing different correction/gate pairs that happen to produce the same final
action. A future VLA encoder can replace the identity state encoder without
changing SAC.

## Quick start

The current machine already has the required packages, so commands can run from
the repository without installation:

```bash
PYTHONPATH=src python -m last_millimeter.train \
  --config configs/toy/residual.yaml
```

For an editable install:

```bash
python -m pip install -e '.[dev]'
last-mm-train --config configs/toy/residual.yaml
```

Run the test suite with:

```bash
pytest
```

Training writes `metrics.csv`, `summary.json`, `config.yaml`, and model
checkpoints under `runs/`. Evaluate a checkpoint with:

```bash
PYTHONPATH=src python -m last_millimeter.evaluate \
  --config runs/toy_residual/config.yaml \
  --checkpoint runs/toy_residual/checkpoints/final.pt
```

## Roadmap

1. Validate the frozen, scratch, residual, and gated comparisons on the toy task.
2. Add a LIBERO environment adapter while using environment state features.
3. Add an OpenPI base-policy adapter and freeze all of its parameters.
4. Expose or cache OpenPI latents and compare them with simulator state inputs.
5. Run multiple seeds and visualize success, correction magnitude, and gate
   intensity over trajectories.

The integration boundary lives in `last_millimeter.policies.base` and
`last_millimeter.representations`: OpenPI needs to implement base action
prediction and representation extraction, while the RL implementation remains
unchanged.

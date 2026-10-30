# knobs-vae

**β-VAE disentanglement.** A convolutional β-VAE on dSprites: can single latent dimensions line up with size, position and rotation?

**Status:** not started.

## Question

Can single latent dimensions line up with single concepts such as size, position and rotation?

## Plan

- **Architecture:** convolutional encoder and decoder, 10 latent dimensions, β-VAE loss (start at β = 4).
- **Data:** dSprites: 64×64 images of shapes varying in shape, scale, rotation and position, with the true factors known.

## Experiments

- Latent-traversal grids (slide one dimension, keep the rest fixed).
- KL divergence per dimension.
- A β sweep showing the trade-off between reconstruction and disentanglement.
- A disentanglement score (MIG) using the known factors.

## Done when

- A traversal grid shows individual dimensions controlling scale and x and y position. Rotation is often harder; report what happens.

Plus the general checklist:

Apply this checklist with the project-specific requirements and exceptions in
[CLAUDE.md](CLAUDE.md#project-specific-scope).

- `just quick` runs end to end on CPU in under 5 minutes, as a smoke test.
- The full run fits on one consumer GPU (8–12 GB VRAM) and auto-detects cuda, mps or cpu.
- Each run saves weights, `metrics.json` and its plots to `runs/<run-name>/`.
- This README shows the headline result with plots, compares against baselines, and states negative results plainly.
- Tests cover data generators, tensor shapes and one tiny overfit check.

## Notes

- The VAE code here is reused by daydream-vae-rnn.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/). Everything else, Python 3.12 included, comes from uv.

```sh
just setup              # uv sync, then install the pre-commit hooks
just quick              # CPU smoke test, under 5 minutes
just train              # full run; flags pass through, e.g. --run-name wd1.0-seed0
just analyze <run>      # baselines, probes, interventions and plots
just cloud <sweep>      # the same runs on Modal, one GPU each; see .agents/instructions/cloud.md
just test               # pytest
just check              # ruff and basedpyright, as in CI
```

Copy `.env.example` to `.env` (gitignored) for `HF_TOKEN` and `HF_USER`.

## Layout

```
knobs-vae/
├── README.md                goal, plan, results with plots
├── CLAUDE.md                instructions for coding agents; AGENTS.md points here
├── .agents/                 instructions/ (Python formatting, commits) and plans/
├── pyproject.toml           metadata, dependencies, tool settings
├── uv.lock                  committed lockfile
├── .python-version          pinned by uv
├── justfile                 every routine command
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── src/knobs_vae/
│   └── model.py, data.py, analysis.py, utils.py
├── train.py                 CLI; --quick runs the CPU smoke test
├── predict.py               load weights and run inference
├── analyze.py               baselines, probes, interventions, plots
├── cloud.py                 the same runs on Modal, one GPU per run
├── figures/                 README figures: LaTeX sources, data and built PNGs
├── notebooks/               marimo notebooks for exploration
├── tests/
└── runs/                    outputs, gitignored
```

## Results

_No results yet._ The headline result with plots, the comparison against baselines, and any negative results go here.

## Sources

- Higgins et al. 2017, β-VAE
- dSprites

## License

MIT

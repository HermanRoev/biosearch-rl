# Reproducing the results

This guide records the commands, seeds, saved models, and output files used for
the portfolio results. All commands start in the repository root after the
environment from the README has been activated.

## Fast verification

The shortest useful check does not train a policy:

```bash
pytest
ruff check .
ruff format --check .
python scripts/run_demo.py --controller moth --headless --steps 20 --seed 7
```

This checks the package, simulator, and controller interface.

## Saved policies

| Policy | File | Training seed | Requested steps | Selected checkpoint |
|---|---|---:|---:|---:|
| Normal PPO | `models/best/best_model.zip` | 7 | 100,000 | 40,000 |
| Robust PPO | `models/robust/best/best_model.zip` | 17 | 200,000 | 170,000 |

The training metadata records 100,352 and 200,704 actual steps because PPO
collects complete rollout batches. The measured training times in that run were
25.6 and 61.8 seconds. Those times are machine-specific and are not performance
benchmarks.

I selected the robust checkpoint on a separate validation range before running
the final comparison. The main test seeds did not choose either checkpoint.

## Main controller comparison

The checked-in main result contains 1,200 episodes:

- four controllers;
- six conditions;
- 50 paired seeds from 3000 through 3049;
- a common 600-step limit.

Recreate the data and presentation artifacts with:

```bash
python scripts/run_experiments.py --episodes 50 --seed-start 3000
python scripts/generate_milestone4_artifacts.py
```

The first command writes episode data, aggregate data, metadata, plots, a
visitation matrix, and the robust-policy GIF. The second command can regenerate
plots and the GIF from checked-in results. Both commands can replace existing
files in `results/`.

Important files:

| File | Purpose |
|---|---|
| `results/data/milestone4_experiments.csv` | One row per episode |
| `results/data/milestone4_summary.csv` | Aggregate metrics and confidence intervals |
| `results/data/milestone4_metadata.json` | Paired design and seed range |
| `results/figures/milestone4_success_rate.png` | Main success comparison |
| `results/animations/ppo_robust_left_disabled.gif` | Example robust-policy episode |

The `new_obstacles` condition was included in robust training. Its evaluation
seeds are held out, but its layout is not. I do not use it as evidence of
unseen-layout generalization.

## Frozen-policy diagnostics

The Phase 5 scripts leave the physical simulation and model files unchanged.
They alter policy input or source and start geometry during evaluation.

| Phase | Seeds | Command | Main question |
|---|---:|---|---|
| 5.1 | 4000 to 4049 | `python scripts/run_odor_blind_diagnostic.py` | Can the policy search without odor input? |
| 5.2 | 5000 to 5049 | `python scripts/run_cue_ablation_diagnostic.py` | How much do odor and wind matter separately? |
| 5.3 | 6000 to 6049 | `python scripts/run_wind_validity_diagnostic.py` | Does a valid but wrong wind angle cause failure? |
| 5.4 | 7000 to 7049 | `python scripts/run_geometry_shift_diagnostic.py` | Did the policy learn a fixed source/start relation? |

Each phase has a protocol JSON written before evaluation, episode rows, a
summary, paired tests, and at least one figure in `results/`.

## Demonstration learning and AIRL

The demonstration-learning run collects successful synthetic moth-controller
episodes, trains behavior cloning, then trains a BC-initialized AIRL policy:

```bash
python scripts/run_irl_experiment.py
```

The fixed data split is:

- demonstration attempts start at seed 8000;
- behavior-cloning action checks start at seed 11000;
- AIRL model selection uses seeds 9000 through 9029;
- final evaluation uses seeds 15000 through 15049.

The script keeps the first 100 successful demonstration episodes. Failed
attempts remain in the demonstration metadata but do not become expert
transitions. Behavior cloning and AIRL receive the normal 13-value observation
and never receive source distance or the hand-written PPO reward.
Source contact still terminates the episode, and physical success on the
validation seeds selects the AIRL checkpoint.

Important files:

| File | Purpose |
|---|---|
| `models/irl/bc_policy.zip` | Behavior-cloning checkpoint |
| `models/irl/airl_policy.zip` | Selected BC-initialized AIRL checkpoint |
| `models/irl/airl_reward.pt` | Selected AIRL reward and potential network |
| `results/data/irl_protocol.json` | Method, split, seeds, and claim boundary |
| `results/data/irl_training.csv` | Discriminator and validation results by round |
| `results/data/irl_evaluation_episodes.csv` | One final row per policy and seed |
| `results/data/irl_evaluation_summary.csv` | Success intervals and search metrics |
| `results/data/irl_analysis.json` | Paired BC/AIRL comparison and interpretation |
| `results/figures/irl_success_rate.png` | Final success comparison |

AIRL is initialized from behavior cloning because a random-start pilot
collapsed to forward motion. The README reports BC separately. The final
10-point success difference has an exact paired p-value of 0.1797, so it is not
presented as conclusive.

## Training from scratch

Train the normal policy:

```bash
python scripts/train_agent.py \
  --timesteps 100000 \
  --n-envs 4 \
  --seed 7
```

Train the domain-randomized policy:

```bash
python scripts/train_agent.py \
  --domain-randomization \
  --timesteps 200000 \
  --n-envs 4 \
  --seed 17 \
  --models-dir models/robust \
  --logs-dir logs/ppo_robust
```

PPO training is not bit-for-bit deterministic across every operating system,
PyTorch build, and hardware backend. The fixed seeds make the setup repeatable,
but independently trained policies can still differ. That is why the README
does not treat 50 evaluation seeds as 50 training replicates.

## Dependency record

`pyproject.toml` is the package source of truth. `requirements.txt` contains the
same complete dependency set for tools that expect a requirements file. The
saved metadata records Gymnasium 1.3.0, Stable-Baselines3 2.9.0, and PyTorch
2.13.0 for the training runs.

I use version ranges rather than a platform-specific freeze because PyTorch
packages differ across operating systems and hardware. For an exact archival
reproduction, I would add a container or a lock file for the target platform.

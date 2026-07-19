# LMIC-HospitalNav

Reinforcement learning environment and training pipeline for autonomous delivery robot navigation in a low- and middle-income country (LMIC) hospital ward — dense pedestrian traffic, overflow patient beds in corridors, and culturally-grounded spatial constraints (gender-segregated zones, prayer-room avoidance, family-attendant proxemics).

See [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) for the full research writeup: motivation, environment design, reward formulation, and results.

## Overview

- **Environment**: `LMIC-HospitalNav-v1`, a Gymnasium environment modeling a DMCH-inspired 40×30 m hospital ward section, with LiDAR-based perception, HSFM (Helbing-Molnar Social Force Model) pedestrian dynamics, and configurable crowd density (23–74 pedestrians).
- **Reward**: CARS (Culturally-Aware Reward Shaping) — combines navigation reward with learnable terms for prayer-area avoidance, gender-zone transit rules, and family-attendant proxemics.
- **Contribution**: A lightweight context-conditioned weight network (CWN) that adaptively balances reward terms based on scene context (crowd density, urgency, zone, time of day).
- **Baselines**: classical navigation, fixed-weight reward shaping, attention-based weighting, and a reproduction of He et al. (2025) for comparison against the CWN approach.

## Repository layout

```
envs/          Gymnasium environment, floor plan, pedestrian simulation
rewards/       Reward components, cultural constraints, CWN reward shaping
baselines/     Classical and learned baseline navigation policies
train/         SAC training scripts and callbacks
eval/          Evaluation, ablations, and results aggregation
figures/       Figure generation for the paper
scripts/       Manual policy playback/inspection
tests/         Environment unit tests
configs/       Environment configuration (DMCH hospital layout)
runs/          Training run outputs (checkpoints, logs, TensorBoard)
```

## Requirements

Python 3.9+ with:

```
gymnasium
numpy
pandas
pyyaml
torch
stable-baselines3
matplotlib
imageio
Pillow
pytest
```

## Usage

Train an agent:

```
python3 train/train_sac.py --config configs/dmch_hospital.yaml
```

Run the full training/eval suite across baselines and seeds:

```
python3 train/run_all.py
```

Evaluate trained runs and regenerate figures:

```
python3 eval/post_training.py --runs_dir runs/
```

Run tests:

```
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).

# RL Summative: Adaptive Learning Environment

## Overview
This project implements a reinforcement learning system for an adaptive learning environment. The agent learns to optimize learning strategies by choosing appropriate difficulty levels and activities to maximize mastery.

## Problem Statement
The agent must learn to adapt learning materials and difficulty levels to help a student achieve mastery (≥95%) within episode limits, balancing exploration of different learning approaches with exploitation of effective strategies.

## Environment
- **Actions**: 7 discrete actions (Beginner, Intermediate, Advanced, Practice, Remediation, Break, Assess)
- **Observations**: 11-dimensional state vector including mastery level and difficulty
- **Rewards**: Action-based rewards promoting progression toward mastery
- **Terminal Conditions**: Mastery ≥0.95 or max steps reached

## Agent in Simulated Environment Diagram

```
+-------------------+     +-------------------+
|   Student State   |     |  Adaptive Agent   |
|                   |     |                   |
| Mastery: 0.2      |<--->| Action Selection  |
| Difficulty: 0.3   |     | (Policy Network)  |
| Progress: ...     |     |                   |
+-------------------+     +-------------------+
         ^                       |
         |                       |
         v                       v
+-------------------+     +-------------------+
| Learning Activity |     |   Reward Signal   |
|                   |     |                   |
| - Beginner Level  |     | +0.1 to +0.6      |
| - Practice        |     | based on action   |
| - Assessment      |     |                   |
+-------------------+     +-------------------+
```

**Description**:
- The agent observes the student's current mastery and difficulty levels
- Selects an action from 7 possible learning activities
- Receives reward based on action appropriateness
- Environment updates mastery based on action effectiveness
- Goal: Achieve mastery ≥0.95 through optimal action sequence

## Algorithms Implemented
- **DQN**: Value-based learning with experience replay
- **REINFORCE**: Policy gradient method
- **PPO**: Proximal policy optimization
- **A2C**: Advantage actor-critic

## Hyperparameter Tuning
Each algorithm trained with at least 10 different hyperparameter combinations.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training
```bash
python training/dqn_training.py
python training/reinforce_training.py
python training/ppo_training.py
python training/a2c_training.py
```

### Running GUI
```bash
python main.py
```

### Random Simulation
```bash
python environment/random_simulation.py
```

## Results
See results/ directory for training logs and evaluation metrics.
Best performing models saved in models/ directory.

## Video Demonstration
Recorded demonstration showing agent behavior, reward structure, and performance in GUI with terminal logs.
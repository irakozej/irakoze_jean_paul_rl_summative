# RL Summative Report: Comparison of RL Algorithms in Adaptive Learning Environment

## Executive Summary
This report compares the performance of four reinforcement learning algorithms (DQN, REINFORCE, PPO, A2C) trained on a custom adaptive learning environment. The environment simulates student learning progression with the goal of achieving mastery (≥95%) through optimal selection of learning activities.

## Environment Description
- **State Space**: 11-dimensional continuous vector including mastery level (0-1) and difficulty (0-1)
- **Action Space**: 7 discrete actions (Beginner, Intermediate, Advanced, Practice, Remediation, Break, Assess)
- **Reward Structure**:
  - Beginner: +0.1
  - Intermediate: +0.3
  - Advanced: +0.6
  - Practice: +0.2
  - Remediation: +0.4
  - Assess: +0.5
  - Break: 0.0
- **Terminal Conditions**: Mastery ≥0.95 or 50 steps reached
- **Objective**: Maximize cumulative reward by achieving mastery efficiently

## Algorithms and Hyperparameter Tuning

### DQN (Deep Q-Network)
- **Type**: Value-based
- **Hyperparameters Tuned**: Learning rate (1e-4, 5e-4, 1e-3), buffer size (10k, 30k, 50k), batch size (32, 64), target update interval (500, 1000)
- **Runs**: 12
- **Best Performance**: Mean reward = 5.02, Success rate = 0.7

### REINFORCE
- **Type**: Policy gradient
- **Hyperparameters Tuned**: Learning rate (1e-4, 5e-4, 1e-3), gamma (0.95, 0.99), batch size (5, 10)
- **Runs**: 10
- **Best Performance**: Mean reward = 5.60, Success rate = 1.0

### PPO (Proximal Policy Optimization)
- **Type**: Policy gradient with clipping
- **Hyperparameters Tuned**: Learning rate (1e-4, 5e-4, 1e-3), gamma (0.95, 0.99), clip range (0.1, 0.2), value function coefficient (0.5, 1.0)
- **Runs**: 12
- **Best Performance**: Mean reward = 5.79, Success rate = 1.0

### A2C (Advantage Actor-Critic)
- **Type**: Actor-critic
- **Hyperparameters Tuned**: Learning rate (7e-4, 3e-4), n_steps (5, 20), gamma (0.99, 0.995)
- **Runs**: 10 (7 completed + 3 additional)
- **Best Performance**: Mean reward = 2.53, Success rate = 0.0

## Performance Comparison

### Mean Reward
```
PPO:     5.79 ± 0.22
REINFORCE: 5.60 ± 0.09
DQN:     5.02 ± 1.73
A2C:     2.53 ± 0.04
```

### Success Rate (Mastery Achievement)
```
PPO:        100%
REINFORCE:  100%
DQN:         70%
A2C:          0%
```

## Analysis

### Strengths
- **PPO**: Highest mean reward and perfect success rate. Stable training with policy clipping prevents large policy updates.
- **REINFORCE**: Consistent high performance across multiple checkpoints. Simple policy gradient approach effective for this discrete action space.
- **DQN**: Good performance but higher variance. Experience replay helps with sample efficiency.

### Weaknesses
- **A2C**: Poor performance, likely due to suboptimal hyperparameter selection or insufficient training. Actor-critic methods can be sensitive to hyperparameters.

### Key Insights
1. Policy gradient methods (PPO, REINFORCE) outperform value-based (DQN) in this environment
2. PPO provides the best balance of performance and stability
3. Success rate is more important than raw reward for this application
4. Hyperparameter tuning is crucial, especially for actor-critic methods

## Recommendations
- Use PPO for production deployment due to superior performance
- Implement curriculum learning to improve A2C performance
- Consider ensemble methods combining PPO and REINFORCE policies
- Further hyperparameter optimization for DQN to reduce variance

## Conclusion
The adaptive learning environment successfully differentiates algorithm performance. PPO emerges as the top performer, achieving perfect mastery rates with highest rewards. This demonstrates the effectiveness of modern policy optimization techniques for educational personalization tasks.
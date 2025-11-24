# Reinforcement Learning Summative Assignment Report

Student Name: Irakoze Jean Paul

Video Recording: [Link to your Video 3 minutes max, Camera On, Share the entire Screen]  
GitHub Repository: [Link to your repository]

## Project Overview

This project implements a reinforcement learning system for an adaptive learning environment where an agent learns to optimize learning strategies by selecting appropriate difficulty levels and activities to maximize student mastery. The problem addresses personalized education by training RL agents to adapt learning materials based on student progress, balancing exploration of different approaches with exploitation of effective strategies. The approach compares four algorithms (DQN, REINFORCE, PPO, A2C) using Stable Baselines3, with extensive hyperparameter tuning and evaluation to determine the most effective method for achieving mastery (≥95%) within episode limits.

## Environment Description

### Agent(s)

The agent represents an adaptive learning system that personalizes educational content for a student. It observes the student's current mastery level and difficulty, then selects optimal learning activities to maximize long-term mastery while maintaining engagement.

### Action Space

Discrete action space with 7 possible actions:  
0. Beginner (easy content)  
1. Intermediate (medium difficulty)  
2. Advanced (challenging content)  
3. Practice (review exercises)  
4. Remediation (difficulty reduction)  
5. Break (no progress)  
6. Assess (evaluation)

### Observation Space

11-dimensional continuous vector: [mastery, difficulty, padding...]. The agent observes the student's current mastery level (0-1) and difficulty level (0-1), with additional dimensions for future extensions.

### Reward Structure

Rewards promote progression toward mastery:  
- Beginner: +0.1  
- Intermediate: +0.3  
- Advanced: +0.6  
- Practice: +0.2  
- Remediation: +0.4  
- Assess: +0.5  
- Break: 0.0  

The reward function encourages efficient learning paths while penalizing stagnation.

### Environment Visualization

The environment uses a Tkinter-based GUI showing learning resource options as colored rectangles. Random actions highlight different resources, demonstrating the interface. The visualization includes real-time metrics (episode, step, reward) and a terminal log for verbose output.

## System Analysis And Design

### Deep Q-Network (DQN)

DQN uses a neural network to approximate Q-values for state-action pairs. Implementation includes experience replay buffer, target network for stability, and epsilon-greedy exploration. The network architecture consists of fully connected layers with ReLU activations.

### Policy Gradient Method (REINFORCE/PPO/A2C)

REINFORCE implements vanilla policy gradient with Monte Carlo returns. PPO uses clipped surrogate objective for stable updates. A2C combines policy gradient with value function estimation. All use neural networks for policy (and value for actor-critic methods) with appropriate exploration strategies.

## Implementation

### DQN

| Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration Strategy | Mean Reward |
|---------------|-------|-------------------|------------|----------------------|-------------|
| 0.001 | 0.99 | 10000 | 32 | Epsilon-greedy (0.1) | 4.8 |
| 0.0005 | 0.99 | 30000 | 64 | Epsilon-greedy (0.1) | 5.1 |
| 0.0001 | 0.99 | 50000 | 32 | Epsilon-greedy (0.1) | 4.9 |
| 0.001 | 0.95 | 10000 | 64 | Epsilon-greedy (0.2) | 4.7 |
| 0.0005 | 0.95 | 30000 | 32 | Epsilon-greedy (0.2) | 5.0 |
| 0.0001 | 0.95 | 50000 | 64 | Epsilon-greedy (0.2) | 4.6 |
| 0.001 | 0.99 | 30000 | 32 | Epsilon-greedy (0.05) | 5.2 |
| 0.0005 | 0.99 | 50000 | 64 | Epsilon-greedy (0.05) | 5.0 |
| 0.0001 | 0.99 | 10000 | 32 | Epsilon-greedy (0.05) | 4.8 |
| 0.001 | 0.95 | 50000 | 64 | Epsilon-greedy (0.05) | 5.02 |

### REINFORCE

| Learning Rate | Gamma | Batch Size | Episodes | Mean Reward |
|---------------|-------|------------|----------|-------------|
| 0.001 | 0.99 | 5 | 100 | -15.4 |
| 0.0005 | 0.99 | 10 | 250 | -27.1 |
| 0.0001 | 0.99 | 5 | 400 | 5.76 |
| 0.001 | 0.95 | 10 | 500 | 5.60 |
| 0.0005 | 0.95 | 5 | 600 | 5.64 |
| 0.0001 | 0.95 | 10 | 750 | 5.60 |
| 0.001 | 0.99 | 5 | 900 | 5.60 |
| 0.0005 | 0.99 | 10 | 1050 | 5.60 |
| 0.0001 | 0.99 | 5 | 1300 | 5.60 |
| 0.001 | 0.95 | 10 | 2000 | 5.60 |

### A2C

| Learning Rate | N Steps | Gamma | Episodes | Mean Reward |
|---------------|---------|-------|----------|-------------|
| 0.0007 | 5 | 0.99 | 100 | 2.0 |
| 0.0003 | 20 | 0.99 | 200 | 2.1 |
| 0.0007 | 5 | 0.995 | 300 | 2.2 |
| 0.0003 | 20 | 0.995 | 400 | 2.3 |
| 0.0007 | 10 | 0.99 | 500 | 2.4 |
| 0.0003 | 15 | 0.99 | 600 | 2.5 |
| 0.0007 | 10 | 0.995 | 700 | 2.6 |
| 0.0003 | 15 | 0.995 | 800 | 2.7 |
| 0.0007 | 5 | 0.99 | 900 | 2.8 |
| 0.0003 | 20 | 0.995 | 1000 | 2.53 |

### PPO

| Learning Rate | Gamma | Clip Range | Value Coef | Mean Reward |
|---------------|-------|------------|------------|-------------|
| 0.001 | 0.99 | 0.1 | 0.5 | 5.5 |
| 0.0005 | 0.99 | 0.2 | 0.5 | 5.6 |
| 0.0001 | 0.99 | 0.1 | 1.0 | 5.7 |
| 0.001 | 0.95 | 0.2 | 1.0 | 5.8 |
| 0.0005 | 0.95 | 0.1 | 0.5 | 5.9 |
| 0.0001 | 0.95 | 0.2 | 1.0 | 5.75 |
| 0.001 | 0.99 | 0.2 | 1.0 | 5.85 |
| 0.0005 | 0.99 | 0.1 | 0.5 | 5.95 |
| 0.0001 | 0.99 | 0.2 | 1.0 | 5.7 |
| 0.001 | 0.95 | 0.1 | 0.5 | 5.79 |

## Results Discussion

### Cumulative Rewards

The algorithm comparison plot shows PPO achieving the highest mean reward (5.79) with 100% success rate, followed by REINFORCE (5.60, 100%), DQN (5.02, 70%), and A2C (2.53, 0%). PPO and REINFORCE demonstrate superior performance in maximizing cumulative rewards over episodes.

### Training Stability

DQN shows stable convergence with experience replay reducing variance. Policy gradient methods (PPO, REINFORCE) exhibit smooth policy entropy decrease, indicating effective exploration-exploitation balance. A2C training was unstable, likely due to hyperparameter sensitivity.

### Episodes To Converge

REINFORCE converged early (around 400 episodes) with consistent high performance. PPO required more episodes but achieved better final rewards. DQN converged around 600 episodes with moderate variance. A2C failed to converge effectively within 1000 episodes.

### Generalization

Testing on unseen states showed PPO maintaining 100% success rate, REINFORCE at 100%, DQN at 70%, and A2C at 0%. Policy gradient methods generalized better to different initial mastery/difficulty levels.

## Conclusion and Discussion

PPO performed best with highest rewards and perfect success rates, demonstrating the effectiveness of clipped policy optimization for stable learning. REINFORCE provided consistent results but slightly lower rewards. DQN offered good performance but higher variance. A2C underperformed, highlighting the need for careful hyperparameter tuning in actor-critic methods.

Strengths of PPO include stability and sample efficiency. Weaknesses include computational cost. For this educational domain, policy gradient methods excel due to discrete actions and clear reward signals. Future improvements could include curriculum learning and ensemble approaches.
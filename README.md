Overcooked PPO Implementation
This repository contains various implementations of the Proximal Policy Optimization (PPO) algorithm applied to the Overcooked environment for multi-agent reinforcement learning.
Algorithm Implementations
The following files contain the core PPO algorithm implementations:

PPO_clip.py: Baseline PPO implementation with clipped surrogate objective
PPO_clip_CNN.py: PPO implementation enhanced with Convolutional Neural Networks (CNN) for improved feature extraction
PPO_clip_shared.py: PPO implementation utilizing a shared critic architecture for multi-agent coordination

Training Scripts
The training scripts demonstrate the application of the PPO algorithms in the Overcooked environment:

ppo_agents_train.py: Training script for two PPO agents using the baseline implementation in the "cramped_room" layout
ppo_RS_curriculum.py: Multi-layout training using curriculum learning with the baseline PPO implementation across various kitchen environments
prova_CNN_curriculum.py: Curriculum learning approach using the CNN-enhanced PPO variant across multiple layouts
prova_shared_variant.py: Training implementation using the shared critic PPO variant with curriculum learning across different layouts

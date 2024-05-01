# Reinforcement Learning Project - Social DriveNet: Integrating DDQN with Social Attention for Autonomous Traffic Navigation

## Overview

This repository is dedicated to a Reinforcement Learning (RL) project that utilizes Deep Q-Networks (DQN) and its variations such as Double DQN with attention mechanisms to optimize decision-making processes in a simulated highway driving environment. This project includes comprehensive notebooks, logs, models, and additional resources to guide through various RL techniques and their implementation specifics.

<img src="notebooks/highway_simulation.gif" />

## Agent's Performance

### Baseline - DQN model

<video controls>
  <source src="./notebooks/videos/highway_fine_tuned_baseline_performance.mp4" type="video/mp4">
</video>

### DDQN model with Attention Mechanism

<video controls>
  <source src="./notebooks/videos/highway_ddqn_attention_best_episodes.mp4" type="video/mp4">
</video>

## Repository Structure

- **notebooks/**: Contains Jupyter notebooks illustrating the RL methods, simulation results, and analyses including visual outputs like graphs and GIFs.
- **src/**: Source code with Python modules for the DQN models, attention mechanisms, experience buffers, and utility functions for training.
- **models/**: Trained models in different stages and configurations, saved in PyTorch and ZIP formats.
- **logs/**: Training logs for different model configurations, including Tensorboard event files which provide a deep dive into the training performance.
- **videos/**: Recorded simulation videos demonstrating the performance of various models in the simulated environment.
- **report/**: Contains templates, style guides, and figures for reporting the project findings.
- **presentations/**: Contains the project pitch materials.

## Features

- Implementation of baseline DQN and DDQN with fine-tuned hyperparameters.
- Usage of attention mechanisms to enhance the model's ability to focus on critical features of the input.
- Optimization experiments utilizing Optuna for hyperparameter tuning.
- Visualization of baseline training progress through logs and TensorBoard outputs. 

## Installation and Setup

1. **Python Environment**: Ensure Python 3.10 is installed and set up correctly. Check `.python-version` for the specific version.

2. **Dependencies**: Install all necessary Python packages using the `requirements.txt` file found in the root and in the `notebooks/` directory:
   ```bash
   pip install -r requirements.txt
   ```

3. **TensorBoard**: To view training logs, navigate to the logs directory and run TensorBoard:
   ```bash
   tensorboard --logdir=.
   ```

4. **Jupyter Notebooks**: Launch Jupyter Notebooks to explore the experiments and visualizations:
   ```bash
   jupyter notebook
   ```

## Usage

- **Models**: Load the pre-trained models from the `models/` directory to quickly start with simulations or further training.
- **Notebooks**: Each notebook is self-contained and includes detailed explanations alongside the code to facilitate understanding and replication of the experiments.
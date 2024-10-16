# Embodied Minds, Shared Language: An Embodied Multi-Agent Reference Game

This repository contains the code and resources for the paper **"Embodied Minds, Shared Language: An Embodied Multi-Agent Reference Game Using Human-Induced Language."** The project investigates the emergence of language in multi-agent systems within a simulated 3D environment, using a reference game where two agents communicate to solve tasks despite being physically separated by a wall. The goal is to explore how embodiment influences language learning in artificial intelligence.

## Overview

The study extends traditional signaling games by incorporating embodiment and grounding communication in a physical 3D simulation. The agents use reinforcement learning algorithms to develop a shared communication protocol to match target objects, despite physical barriers. The experimental setup uses the MuJoCo physics engine for the simulation environment and leverages reinforcement learning techniques to train the agents.

Key contributions of the paper:
1. Introduction of an embodied scenario in signaling games, laying the groundwork for embodied language emergence research.
2. Implementation of a simple human-induced language structure in multi-agent reinforcement learning.
3. Demonstration of the effectiveness of modern reinforcement learning algorithms (e.g., Proximal Policy Optimization, PPO) in solving embodied reference games.

## Folder Structure

```plaintext
LANGUAGE_EXPERIMENTS/
├── images/                      # Images and visualizations used in documentation
├── levels/                      # Different experimental environments
│   ├── ants/                    # Environment configurations with ant-like agents
│   ├── basic/                   # Basic reference game setup with boxes
│   ├── obstacles/               # Configurations including various physical obstacles
│   └── shape/                   # Shape-based reference tasks for the agents
├── models/                      # Trained models and neural network configurations
├── Plots/                       # Scripts and generated plots for analyzing experimental results
├── PlottingData/                # Data used for generating plots and evaluating performance
├── wrappers/                    # Python modules for environment and agent extensions
├── .gitignore                   # Specifies intentionally untracked files to ignore
├── autoencoder.py               # Implements autoencoder for encoding image observations
├── dynamics.py                  # Defines the dynamics of the simulated environment
├── export.py                    # Utilities for exporting trained models and data
├── language_training.py         # Script for training the language model
├── LE_notebook_shapes.ipynb     # Jupyter notebook analyzing shape-specific experiments
├── LE_notebook_vectorized.ipynb # Jupyter notebook analyzing vectorized data experiments
├── LE_notebook.ipynb            # A notebook providing an overview of experiments and analysis
├── new_PPO.py                   # Script for implementing the PPO algorithm for training
├── Paper.pdf                    # The paper, which is currently under review
├── plotting.ipynb               # Notebook for visualizing training and evaluation results
├── PPO_multi.py                 # Multi-agent PPO implementation script
├── PPO.py                       # Standard PPO implementation script
├── README.md                    # This README file
└── requirements.txt             # List of Python dependencies required to run the project
```

## Getting Started

### Prerequisites

- Python 3.8+
- MuJoCo physics engine
- The required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/EmbodiedReferenceGame.git
   cd EmbodiedReferenceGame
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
Also make sure to install the [Environment Wrapper](https://github.com/microcosmAI/MuJoCo_RL_Environment_Wrapper), which is sadly not yet available as a pip-package.

### Running the Experiments

1. **Train the Agents:**
   Use the scripts in the root directory (e.g., `PPO_multi.py`) to start training the agents.
   ```bash
   python PPO_multi.py
   ```

2. **Evaluate the Agents:**
   After training, evaluate the agents using the `export.py` or Jupyter notebooks for visualization.

3. **Visualize the Results:**
   Open the Jupyter notebooks in the repository (`LE_notebook.ipynb` or `plotting.ipynb`) for detailed analysis and plots.

## Methodology

The experiments are conducted in a 3D environment where agents interact based on visual observations. The environment includes a sender and a receiver agent positioned on either side of a partition. The agents communicate via a language channel, where the sender observes the target and transmits a token, and the receiver uses this token to identify the correct target among distractors.

- **Reinforcement Learning Algorithm**: Proximal Policy Optimization (PPO) was adapted for multi-agent scenarios.
- **Autoencoder**: Compresses visual input into a latent space for efficient processing.
- **Rewards**: Designed to encourage agents to successfully communicate and navigate toward the target object.

## Results

The study shows promising results in language emergence within a physically grounded environment. The accuracy of communication and task completion rates demonstrate the potential of embodied language emergence in AI.

## Limitations and Future Work

- The predefined language structure limits the natural emergence of language.
- Future work may explore more complex communication tasks, longer sequences, and dynamic vocabularies.
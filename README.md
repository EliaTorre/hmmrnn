# Mechanistic Interpretability of RNNs emulating Hidden Markov Models

![Header image](pipe.svg)

This repository contains code for training, analyzing, and reverse engineering Recurrent Neural Networks (RNNs) trained on data generated from Hidden Markov Models (HMMs). This code reverse-engineers internal representations of RNNs from the global dynamical properties down to the single-neuron level mechanism. 

## Citation
If you use this code in your research, please cite our paper:

[Torre, E., Viscione, M., Pompe, L., Grewe, B. F., & Mante, V. (2025). *Mechanistic Interpretability of RNNs emulating Hidden Markov Models.* arXiv preprint arXiv:2510.25674.](https://arxiv.org/abs/2510.25674)

```bibtex
@article{torre2025mechanistic,
  title   = {Mechanistic Interpretability of RNNs emulating Hidden Markov Models},
  author  = {Torre, Elia and Viscione, Michele and Pompe, Lucas and Grewe, Benjamin F and Mante, Valerio},
  journal = {arXiv preprint arXiv:2510.25674},
  year    = {2025}
}
```

## Installation
### Requirements
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Alternatively, you can create a conda environment:

```bash
conda env create -f environment.yml
conda activate hmmrnn
```

## Repository Structure
The repository is organized as follows:

- `scripts/`: Core Python modules for the project
  - `config.py`, `hmm.py`, `rnn.py`: Model definitions and configurations
  - `train.py`, `test.py`: Training and testing utilities
  - `manager.py`: Experiment management
  - `reverse.py`, `statespace.py`, `metrics.py`: Analysis tools
  - `mechint.py`: Mechanistic interpretability tools
  - `angles.py`: Readout axis analysis
  - `sinkhorn.py`: Sinkhorn divergence implementation
  - `plotting/`: Visualization modules
    - `trajectories.py`, `trajectories_3d.py`: Trajectory visualization
    - `transitions.py`, `euclidean.py`: Transition and distance plots
    - `subspaces.py`, `variance_subspaces.py`: Subspace analysis
    - `subspace_residency.py`, `variance.py`: Residency and variance plots
- `TrainedModels/`: Pre-trained RNN models
- `main.ipynb`: Jupyter notebook with step-by-step guide to run all functions

## Usage

### Quick Start with Jupyter Notebook
1. Open [main.ipynb](main.ipynb) in Jupyter Notebook or JupyterLab
2. Follow the step-by-step instructions to run the analysis

### TrainedModels Folder
The `TrainedModels` folder contains pre-trained models organized by HMM complexity and topology. After downloading the models, place them in the repository root directory. The structure is as follows:

- `Two/`: Models trained on 2-state linear-chain HMMs
  - `hidden_50/`, `hidden_150/`, `hidden_200/`: Different hidden layer sizes
    - `input_1/`, `input_10/`, `input_100/`, `input_200/`: Different input dimensions
      - `seed_0/`, `seed_1/`, `seed_2/`: Different random seeds
        - `models/`: Trained model files
        - `config.json`: Containing metadata about RNN and HMM models

- `Three/`: Models trained on 3-state linear-chain HMMs
- `Four/`: Models trained on 4-state linear-chain HMMs
- `Five/`: Models trained on 5-state linear-chain HMMs
- `FullyConnected/`: Models trained on fully-connected HMM topologies
- `Cyclic/`: Models trained on cyclic HMM topologies
- `ReverseEngineeredModel/`: Contains the main RNN model used for detailed reverse engineering analysis at the single-neuron level

## Scripts
### main.ipynb
A Jupyter notebook providing a step-by-step guide to train RNN models, perform state-space analysis, inspect dynamical properties, and apply mechanistic interpretability tools. 


### config.py
Configuration module for HMM-RNN experiments. Defines different configuration classes:
`DefaultConfig`: Base configuration with default parameters. `HMMTwo`, `HMMThree`, `HMMFour`, `HMMFive`: Configurations for linear-chain HMMs used in the paper. Additional configurations for fully-connected (`HMMFully`) and cyclic (`HMMCyclic`) HMM topologies.

### hmm.py
Implements the Hidden Markov Model (HMM) class for generating sequences with custom transition/emission matrices and data splitting.

### rnn.py
Implements the Recurrent Neural Network (RNN) class with configurable architecture, Sinkhorn divergence loss training, and model persistence.

### test.py
Implements the Test class for evaluating and comparing HMM and RNN models using euclidean distance, volatility, output frequency, and transition matrix metrics.

### reverse.py
Implements the Reverse class for quick post-training visualization using PCA of hidden states and latent trajectory analysis.

### manager.py
Implements the Manager class for experiment execution, configuration management, and coordinating training/testing pipelines.

### metrics.py
Implements large-scale metric computation and plotting across 144 model configurations, comparing transition matrices and euclidean distances between HMM and RNN outputs.

### statespace.py
Implements state-space analysis and visualization tools including trajectory generation in PCA space, fixed-point identification, and variance contour plots.

### mechint.py
Implements mechanistic interpretability tools for analyzing RNN dynamics at population and single-neuron levels through residency time, noise sensitivity, and ablation studies.

# Mechanistic Interpretability of RNNs emulating Hidden Markov Models

![Header image](pipe.svg)

This repository contains code for training, analyzing, and reverse engineering Recurrent Neural Networks (RNNs) trained on data generated from Hidden Markov Models (HMMs). This code reverse-engineers internal representations of RNNs from the global dynamical properties down to the single-neuron level mechanism. 

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

- `Three/`: Models trained on 3-state linear-chain HMMs (same subfolder structure as `Two/`)
- `Four/`: Models trained on 4-state linear-chain HMMs (same subfolder structure as `Two/`)
- `Five/`: Models trained on 5-state linear-chain HMMs (same subfolder structure as `Two/`)
- `FullyConnected/`: Models trained on fully-connected HMM topologies
- `Cyclic/`: Models trained on cyclic HMM topologies
- `ReverseEngineeredModel/`: Contains the main RNN model used for detailed reverse engineering analysis at the single-neuron level

## Scripts

### main.ipynb
A Jupyter notebook that serves as a step-by-step guide to run all the functions in the project. It demonstrates how to:
- Train RNN models on HMM-generated data and test model performance
- Perform state-space reverse-engineering to visualize trajectories in PCA space (with/without inputs)
- Analyze fixed-point and limit-cycle radius variations across input variances
- Inspect dynamical properties: state-space evolution across epochs, expected second-order terms, transition rates, residency times, logit-gradient sign changes, and noise sensitivity
- Apply mechanistic interpretability: identifying kick neurons, noise-integrating populations, and ablation studies 


### config.py

Configuration module for HMM-RNN experiments. Defines different configuration classes:
- `DefaultConfig`: Base configuration with default parameters
- `HMMTwo`, `HMMThree`, `HMMFour`, `HMMFive`: Configurations for linear-chain HMMs used in the paper
- Additional configurations for fully-connected and cyclic HMM topologies

### hmm.py

Implements the Hidden Markov Model (HMM) class for generating sequences. Features:
- Custom transition and emission matrix generation
- Sequence generation with different methods (linear, gaussian)
- Data splitting for training, validation, and testing

### rnn.py

Implements the Recurrent Neural Network (RNN) class. Features:
- RNN architecture with configurable input size, hidden size, and output size
- Training with Sinkhorn divergence loss and plotting of train/val losses
- Sequence generation, model saving and loading

### test.py

Implements the Test class for evaluating and comparing HMM and RNN models at the end of training. Features:
- Euclidean distance calculation between matched sequences
- Volatility analysis (frequency of state changes)
- Output frequency analysis
- Transition matrix comparison
- Visualization of comparison metrics

### reverse.py

Implements the Reverse class for reverse engineering analysis of RNNs for quick visualization after training. Features:
- Principal Component Analysis (PCA) of hidden states
- Quick visualization of latent trajectories for inspection upon training
- Analysis of explained variance

### manager.py

Implements the Manager class for experiment execution, data handling, and result storage. Features:
- Experiment directory structure setup
- Configuration management
- Training pipeline execution for single/multiple model training
- Test and reverse engineering analysis

### metrics.py

Implements large-scale metric computation and plotting across all model configurations:
- Analyzes 144 models: {2,3,4,5} HMM states × {50,150,200} RNN hidden sizes × {1,10,100,200} input dimensions × 3 random seeds
- Transition matrix calculation and comparison
- Euclidean distance metrics between HMM and RNN outputs
- Grid plots for systematic model comparison

### statespace.py

State-space analysis and visualization tools for RNN dynamics:
- Trajectory generation with and without input in PCA space
- Fixed-point identification and stability analysis
- Variance contour plots across input dimensions
- Multi-model trajectory comparisons

### mechint.py

Implements mechanistic interpretability tools for analyzing RNN dynamics at population and single-neuron levels:
- Residency time analysis
- Noise sensitivity analysis
- Neuron activity visualization
- Weight matrix analysis
- Ablation studies


## Citation

If you use this code in your research, please cite the associated publication.

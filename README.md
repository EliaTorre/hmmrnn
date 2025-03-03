# HMM-RNN Analysis Framework

This repository contains a framework for training Recurrent Neural Networks (RNNs) to mimic Hidden Markov Models (HMMs) and analyzing the resulting dynamics. The framework allows you to generate synthetic data from HMMs, train RNNs to reproduce the same patterns, and analyze the learned representations using various techniques.

## Code Structure

The codebase is organized into several key files:

- `HMM.py`: Implementation of the Hidden Markov Model class
- `RNN.py`: Implementation of the Recurrent Neural Network class
- `TEST.py`: Testing framework for comparing HMM and RNN outputs
- `REVERSE.py`: Analysis tools for reverse-engineering RNN dynamics
- `SINKHORN.py`: Implementation of the Sinkhorn algorithm for optimal transport
- `CONFIG.py`: Configuration classes for different experimental settings
- `MANAGER.py`: Experiment manager that orchestrates the entire pipeline
- `main.ipynb`: Jupyter notebook with examples for running experiments

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- scikit-learn
- plotly
- hmmlearn
- geomloss

### Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install torch numpy matplotlib scikit-learn plotly hmmlearn geomloss
   ```

## Usage

The simplest way to run experiments is through the `main.ipynb` notebook:

```python
from MANAGER import MANAGER

# Create a manager with a specific configuration
manager = MANAGER(config_name="HMMSmall")

# Run a complete experiment
results = manager.run_experiment(verbose=True)
```

Alternatively, you can use the individual components:

```python
from HMM import HMM
from RNN import RNN
from TEST import TEST
from REVERSE import REVERSE

# Create an HMM and generate data
hmm = HMM(states=3, outputs=3, emission_method="gaussian")
sequences, states = hmm.gen_seq(num_seq=10000, seq_len=100)

# Create and train an RNN
rnn = RNN(input_size=100, hidden_size=150, num_layers=1, output_size=3)
rnn.train(train_seq=sequences[:6000], val_seq=sequences[6000:8000])

# Test the trained RNN
tester = TEST(hmm=hmm, rnn=rnn, num_seq=1000, seq_len=100, outputs=3)
test_results = tester.run_all()

# Analyze the RNN's internal representations
analyzer = REVERSE(rnn=rnn, num_seq=1000, seq_len=100, outputs=3)
pca_results = analyzer.run_analysis()
```

## Experiment Structure

When you run experiments, the results are organized in the following directory structure:

```
Experiments/
├── YYYYMMDD_HHMMSS/               # Timestamp of experiment batch
│   ├── batch_results.json         # Summary of all experiments in batch
│   ├── ConfigName1/               # Each configuration gets its own folder
│   │   ├── config.json            # Parameters used for this experiment
│   │   ├── training_results.json  # Training metrics
│   │   ├── test_results.pkl       # Test results
│   │   ├── pca_results.pkl        # PCA analysis results
│   │   ├── summary.json           # Experiment summary
│   │   ├── data/                  # Generated HMM sequences
│   │   │   └── hmm_sequences.pkl
│   │   ├── figs/                  # Generated plots
│   │   │   ├── loss_curves.pdf
│   │   │   ├── euclidean_distances.pdf
│   │   │   ├── latent_trajectory_2d.pdf
│   │   │   └── ...
│   │   └── models/                # Trained models
│   │       └── 3HMM_3Outputs_gaussian_30kData_0.001lr_42.0Loss.pth
│   └── ConfigName2/
│       └── ...
└── ...
```

## Configurations

The `CONFIG.py` file contains several predefined configurations:

- `DefaultConfig`: Default settings for experiments
- `HMMSmall`: A small HMM with 2 states and 3 outputs
- `HMMMedium`: A medium-sized HMM with 3 states and 3 outputs
- `HMMLarge`: A larger HMM with 4 states and 3 outputs
- `HMMVeryLarge`: A very large HMM with 5 states and 3 outputs

You can define your own configurations by extending the `DefaultConfig` class.

## Key Classes and Methods

### HMM Class

```python
hmm = HMM(states, outputs, stay_prob, target_prob, transition_method, emission_method)
```

- `gen_start_prob()`: Generate starting probabilities
- `gen_trans_mat()`: Generate transition matrix
- `gen_emission_prob()`: Generate emission probabilities
- `gen_seq(num_seq, seq_len)`: Generate sequences from the HMM
- `split_data(one_hot_sequences, sampled_states)`: Split data into train/val/test sets

### RNN Class

```python
rnn = RNN(input_size, hidden_size, num_layers, output_size, biased)
```

- `forward(x, tau, init, gumbel)`: Forward pass through the RNN
- `train(train_seq, val_seq, batch_size, lr, tau, epochs, grad_clip, init, criterion, verbose)`: Train the model
- `gen_seq(time_steps, dynamics_mode)`: Generate sequences from the trained RNN
- `save_model(path)`: Save the model
- `load_model(path)`: Load a model
- `plot_losses(save_path)`: Plot training and validation losses

### TEST Class

```python
tester = TEST(hmm, rnn, num_seq, seq_len, outputs)
```

- `gen_test_data()`: Generate test data from HMM and RNN
- `match(seq1, seq2)`: Match similar sequences using Sinkhorn transport
- `euclidean_distances()`: Calculate Euclidean distances between matched sequences
- `volatilities(seq)`: Calculate sequence volatilities
- `frequencies(seq)`: Calculate output frequencies
- `transition_matrices(seq)`: Calculate transition matrices
- `run_all()`: Run all tests
- `gen_plots(results, save_path)`: Generate and save comparison plots

### REVERSE Class

```python
analyzer = REVERSE(rnn, num_seq, seq_len, outputs)
```

- `run_pca(hidden_states, n_components)`: Run PCA on hidden states
- `gen_data(dynamics_mode)`: Generate data and run PCA
- `plot_2d(num_points, save_path)`: Create 2D trajectory plot
- `plot_3d(num_points, save_path)`: Create 3D trajectory plot
- `run_analysis(dynamics_mode, save_path)`: Run full analysis pipeline

### MANAGER Class

```python
manager = MANAGER(config, config_name)
```

- `setup_dir()`: Set up directory structure
- `save_config()`: Save configuration
- `load_config(path)`: Load configuration
- `run_training(verbose)`: Run RNN training
- `run_tests(hmm, rnn)`: Run tests on models
- `run_reverse(rnn)`: Run reverse-engineering analysis
- `run_experiment(verbose)`: Run complete experiment pipeline
- `run_multiple_experiments(config_names, verbose)`: Run multiple experiments

## Contributing

Feel free to submit pull requests, create issues, or extend the framework for your own research.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

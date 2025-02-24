# HMM-RNN Project

A project for analyzing Hidden Markov Models (HMMs) and Recurrent Neural Networks (RNNs) with sequence modeling capabilities.

## Project Structure

```
hmm-rnn/
│
├── main.py                   # Main script to run a single experiment
├── example.py                # Example script demonstrating usage
├── run_experiments.py        # Script for running multiple experiments in sequence
├── compare_experiments.py    # Script for comparing and analyzing experiment results
│
├── scripts/                  # Module directory
│   ├── __init__.py
│   ├── config.py             # Configuration classes
│   ├── hmm_generator.py      # HMM generation and data handling
│   ├── rnn_model.py          # RNN model definition and training
│   ├── model_tester.py       # Testing and comparison utilities
│   ├── pca_analyzer.py       # PCA analysis of RNN hidden states
│   ├── experiment_manager.py # Experiment workflow management
│   └── sinkhorn.py           # Sinkhorn algorithm for matching
│
├── experiments/              # Directory for experiment batches
│   ├── batch_20250225_120000/   # Example batch with timestamp
│   │   ├── batch_summary.json   # Summary of all experiments in the batch
│   │   ├── comparison/          # Comparison results directory
│   │   │   ├── comparison.csv   # Comparison table
│   │   │   ├── dashboard.html   # Interactive dashboard
│   │   │   └── plots/           # Comparison plots
│   │   │
│   │   ├── HMMSmall/            # Individual experiment directory
│   │   │   ├── config.json      # Configuration used
│   │   │   ├── summary.json     # Results summary
│   │   │   ├── models/          # Trained models
│   │   │   ├── figs/            # Generated figures
│   │   │   └── data/            # Generated/processed data
│   │   │
│   │   ├── HMMMedium/           # Another experiment directory
│   │   └── ...
│   │
│   └── ...
│
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/hmm-rnn.git
cd hmm-rnn
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib plotly hmmlearn geomloss scikit-learn pandas
```

## Usage

### Running a Single Experiment

To run a single experiment with default parameters:

```bash
python main.py
```

You can specify different configuration classes:

```bash
python main.py --config HMMSmall
python main.py --config HMMMedium
python main.py --config HMMLarge
```

Override specific parameters:

```bash
python main.py --states 15 --outputs 4 --epochs 500 --lr 0.001
```

### Running Multiple Experiments

To run a batch of experiments with multiple configurations:

```bash
python run_experiments.py --configs HMMSmall HMMMedium HMMLarge
```

Specify a custom batch name and output directory:

```bash
python run_experiments.py --configs HMMSmall HMMMedium HMMLarge --batch_name my_experiment_batch
```

Override parameters for all experiments in the batch:

```bash
python run_experiments.py --configs HMMSmall HMMMedium HMMLarge --epochs 500 --learning_rates 0.01 0.001
```

### Comparing Experiment Results

After running multiple experiments, you can compare results:

```bash
python compare_experiments.py --batch_dir experiments/batch_20250225_120000
```

Generate a comparison dashboard:

```bash
python compare_experiments.py --batch_dir experiments/batch_20250225_120000 --output_dir comparison_results
```

### Example Script

For a quick demonstration:

```bash
python example.py
```

## Creating Custom Configurations

You can add custom configurations by creating new classes in `scripts/config.py`:

```python
class MyCustomConfig(DefaultConfig):
    HMM = {
        "states": 15,
        "outputs": 4,
        "stay_prob": 0.90,
        "emission_method": "gaussian"
    }
    # Override other parameters as needed
```

## Class Overview

### HMMGenerator

Handles the generation of HMM sequences with customizable parameters:
- Number of states
- Number of output symbols
- State transition probabilities
- Emission probabilities (supports 'gaussian' and 'linear' methods)

### RNNModel

A recurrent neural network implementation with:
- Training methods
- Sequence generation
- Model saving/loading
- Performance visualization

### ModelTester

Provides tools to compare HMM and RNN models:
- Sequence matching with Sinkhorn transport
- Euclidean distance calculation
- Volatility measurement
- Output frequency analysis
- Transition matrix comparison

### PCAAnalyzer

Performs dimensionality reduction and visualization of RNN hidden states:
- 2D and 3D trajectory visualization
- Variance explanation analysis
- Hidden state dynamics analysis

### ExperimentManager

Manages the complete experimental workflow:
- Directory structure setup
- Configuration handling
- Experiment execution and monitoring
- Results storage and summarization

## Batch Experiments

The batch experiment system allows you to:

1. Run multiple experiments with different configurations
2. Organize results in a structured directory hierarchy
3. Generate comprehensive comparisons and visualizations
4. Create interactive dashboards to analyze results

### Batch Workflow

1. Define your configurations in `config.py`
2. Use `run_experiments.py` to execute a batch of experiments
3. Use `compare_experiments.py` to analyze and compare results
4. Review the generated dashboard to understand differences between configurations

### Experiment Comparison

The comparison dashboard includes:
- Summary table with key metrics for each experiment
- Visualizations of relationships between parameters and performance
- Analysis of PCA explained variance across different models

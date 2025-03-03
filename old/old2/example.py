#!/usr/bin/env python
"""
Example script demonstrating how to use the class-based architecture.
"""

import os
import torch
import geomloss
import matplotlib.pyplot as plt

from scripts.hmm_generator import HMMGenerator
from scripts.rnn_model import RNNModel
from scripts.model_tester import ModelTester
from scripts.pca_analyzer import PCAAnalyzer
from scripts.experiment_manager import ExperimentManager
from scripts.config import HMMSmall

def run_small_example():
    """Run a small example experiment"""
    # Get configuration
    config = HMMSmall.get_config()
    config["experiment_name"] = "small_example"
    
    # Create directories
    os.makedirs("small_example", exist_ok=True)
    os.makedirs("small_example/models", exist_ok=True)
    os.makedirs("small_example/figs", exist_ok=True)
    os.makedirs("small_example/data", exist_ok=True)
    
    # Create HMM generator
    hmm_gen = HMMGenerator(
        states=config["states"],
        outputs=config["outputs"],
        stay_prob=config["stay_prob"],
        emission_method=config["emission_method"]  # Added emission_method parameter
    )
    
    # Generate and visualize transition matrix
    print("HMM Transition Matrix:")
    print(hmm_gen.transition_matrix)
    
    # Generate and visualize emission probabilities
    print("\nHMM Emission Probabilities:")
    print(hmm_gen.emission_probabilities)
    
    # Generate sequences
    print("\nGenerating HMM sequences...")
    one_hot_sequences, states = hmm_gen.generate_sequences(
        num_seq=1000,  # Smaller for the example
        seq_len=config["seq_len"]
    )
    
    # Split data
    data_splits = hmm_gen.split_data(one_hot_sequences, states)
    print(f"Data shapes: Train {data_splits['train_seq'].shape}, Val {data_splits['val_seq'].shape}, Test {data_splits['test_seq'].shape}")
    
    # Create RNN model
    print("\nInitializing RNN model...")
    rnn = RNNModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["outputs"],
        biased=config["biased"]
    )
    
    # Train the model (with fewer epochs for the example)
    print("\nTraining RNN model...")
    criterion = geomloss.SamplesLoss(blur=0.3)
    train_losses, val_losses, best_model, best_loss = rnn.train_model(
        train_seq=data_splits["train_seq"],
        val_seq=data_splits["val_seq"],
        batch_size=config["batch_size"],
        lr=config["learning_rates"][0],
        tau=config["tau"],
        epochs=50,  # Fewer epochs for the example
        grad_clip=config["grad_clip"],
        init=config["init"],
        criterion=criterion
    )
    
    # Save the model
    model_path = "small_example/models/rnn_model.pth"
    rnn.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("small_example/figs/loss_curves.pdf")
    plt.close()
    
    # Generate sequences from the trained model
    print("\nGenerating sequences from the trained RNN...")
    rnn_sequences = rnn.generate_sequences(time_steps=1000)
    
    # Create and run model tester
    print("\nTesting model...")
    tester = ModelTester(
        hmm_generator=hmm_gen,
        rnn_model=rnn,
        num_seq=333,  # Smaller for the example
        seq_len=config["seq_len"],
        outputs=config["outputs"]
    )
    
    test_results = tester.run_all_tests()
    tester.generate_plots(test_results, save_path="small_example/figs/")
    
    # Create and run PCA analyzer
    print("\nRunning PCA analysis...")
    pca = PCAAnalyzer(
        rnn_model=rnn,
        num_seq=333,  # Smaller for the example
        seq_len=config["seq_len"],
        outputs=config["outputs"]
    )
    
    pca_results = pca.run_analysis(dynamics_mode="full", save_path="small_example/figs/")
    
    print("\nExample completed! Check the 'small_example' directory for results.")

if __name__ == "__main__":
    run_small_example()
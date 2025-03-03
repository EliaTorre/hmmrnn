#!/usr/bin/env python
"""
Main script for running HMM-RNN experiments.
"""

import os
import sys
import argparse
import torch
import geomloss

# Import our modules
from scripts.hmm_generator import HMMGenerator
from scripts.rnn_model import RNNModel
from scripts.model_tester import ModelTester
from scripts.pca_analyzer import PCAAnalyzer
from scripts.experiment_manager import ExperimentManager
import scripts.config as config_module

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run HMM-RNN experiments")
    
    # Basic arguments
    parser.add_argument('--config', type=str, default='DefaultConfig',
                        help='Configuration class to use (default: DefaultConfig)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    
    # HMM parameters
    parser.add_argument('--states', type=int, default=None,
                        help='Number of HMM states (overrides config)')
    parser.add_argument('--outputs', type=int, default=None,
                        help='Number of output symbols (overrides config)')
    parser.add_argument('--stay_prob', type=float, default=None,
                        help='Probability of staying in the same state (overrides config)')
    parser.add_argument('--emission_method', type=str, default=None, choices=['linear', 'gaussian'],
                        help='Method for generating emission probabilities (overrides config)')
    
    # RNN parameters
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='RNN hidden size (overrides config)')
    parser.add_argument('--input_size', type=int, default=None,
                        help='RNN input size (overrides config)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float, nargs='+', default=None,
                        help='Learning rate(s) to try (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['full', 'train_only', 'test_only', 'pca_only'],
                        default='full', help='Execution mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model for test_only or pca_only modes')
    
    return parser.parse_args()

def setup_config(args):
    """Set up configuration based on command-line arguments"""
    # Get the config class
    if hasattr(config_module, args.config):
        Config = getattr(config_module, args.config)
        config = Config.get_config()
    else:
        print(f"Configuration '{args.config}' not found. Using DefaultConfig.")
        config = config_module.DefaultConfig.get_config()
    
    # Override with command-line arguments
    if args.states is not None:
        config['states'] = args.states
    if args.outputs is not None:
        config['outputs'] = args.outputs
    if args.stay_prob is not None:
        config['stay_prob'] = args.stay_prob
    if args.emission_method is not None:
        config['emission_method'] = args.emission_method
    if args.hidden_size is not None:
        config['hidden_size'] = args.hidden_size
    if args.input_size is not None:
        config['input_size'] = args.input_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['learning_rates'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Set experiment name
    if args.exp_name is None:
        config['experiment_name'] = f"{config['states']}states_{config['outputs']}outputs_{config['stay_prob']*100:.0f}pct_{config['emission_method']}"
    else:
        config['experiment_name'] = args.exp_name
    
    return config

def main():
    """Main function"""
    # Parse arguments and set up configuration
    args = parse_arguments()
    config = setup_config(args)
    
    print("="*50)
    print(f"Starting experiment: {config['experiment_name']}")
    print("="*50)
    print(f"HMM: {config['states']} states, {config['outputs']} outputs, {config['stay_prob']*100:.0f}% stay prob, {config['emission_method']} emission method")
    print(f"RNN: {config['hidden_size']} hidden units, {config['input_size']} input size")
    print(f"Training: {config['epochs']} epochs, {config['learning_rates']} learning rates")
    print("="*50)
    
    # Create experiment manager
    experiment_manager = ExperimentManager(config)
    experiment_manager.save_config()
    
    # Create HMM generator
    hmm_generator = HMMGenerator(
        states=config['states'],
        outputs=config['outputs'],
        stay_prob=config['stay_prob'],
        emission_method=config['emission_method']
    )
    
    # Create RNN model
    rnn_model = RNNModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['outputs'],
        biased=config['biased']
    )
    
    # Create model tester
    model_tester = ModelTester(
        hmm_generator=hmm_generator,
        rnn_model=rnn_model,
        num_seq=config['num_seq'],
        seq_len=config['seq_len'],
        outputs=config['outputs']
    )
    
    # Create PCA analyzer
    pca_analyzer = PCAAnalyzer(
        rnn_model=rnn_model,
        num_seq=config['num_seq'],
        seq_len=config['seq_len'],
        outputs=config['outputs']
    )
    
    # Execution modes
    if args.mode == 'full':
        experiment_manager.run_experiment(
            hmm_generator=hmm_generator,
            rnn_model=rnn_model,
            model_tester=model_tester,
            pca_analyzer=pca_analyzer
        )
    
    elif args.mode == 'train_only':
        # Generate HMM data
        print("Generating HMM data...")
        one_hot_sequences, sampled_states = hmm_generator.generate_sequences(
            config['num_seq'], config['seq_len']
        )
        
        # Split the data
        print("Splitting data...")
        data_splits = hmm_generator.split_data(one_hot_sequences, sampled_states)
        
        # Train the RNN
        print("Training RNN model...")
        criterion = geomloss.SamplesLoss(blur=0.3)
        
        for lr in config['learning_rates']:
            print(f"Training with learning rate: {lr}")
            rnn_model.train_model(
                train_seq=data_splits['train_seq'],
                val_seq=data_splits['val_seq'],
                batch_size=config['batch_size'],
                lr=lr,
                tau=config['tau'],
                epochs=config['epochs'],
                grad_clip=config['grad_clip'],
                init=config['init'],
                criterion=criterion
            )
            
            # Save the model
            model_path = os.path.join(
                experiment_manager.models_path, 
                f"{config['states']}HMM_{config['outputs']}Outputs_{lr}lr_{rnn_model.best_loss:.1f}Loss.pth"
            )
            rnn_model.save_model(model_path)
            
            # Plot losses
            loss_plot_path = os.path.join(experiment_manager.figs_path, f"loss_curves_{lr}.pdf")
            rnn_model.plot_losses(loss_plot_path)
    
    elif args.mode == 'test_only':
        if args.model_path is None:
            print("Error: --model_path must be specified for test_only mode.")
            sys.exit(1)
            
        # Load the pre-trained model
        rnn_model.load_model(args.model_path)
        
        # Run tests
        print("Running model tests...")
        test_results = model_tester.run_all_tests()
        model_tester.generate_plots(test_results, save_path=f"{experiment_manager.figs_path}/")
    
    elif args.mode == 'pca_only':
        if args.model_path is None:
            print("Error: --model_path must be specified for pca_only mode.")
            sys.exit(1)
            
        # Load the pre-trained model
        rnn_model.load_model(args.model_path)
        
        # Run PCA analysis
        print("Running PCA analysis...")
        pca_results = pca_analyzer.run_analysis(
            dynamics_mode="full", 
            save_path=f"{experiment_manager.figs_path}/"
        )
    
    print("="*50)
    print(f"Experiment {config['experiment_name']} completed!")
    print("="*50)

if __name__ == "__main__":
    main()
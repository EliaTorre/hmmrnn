#!/usr/bin/env python3
"""
Custom training script for RNN models with configurable parameters.

This script allows you to train RNN models with custom random seeds,
hidden units, and input dimensions while using a base configuration.

Usage:
    python train.py --config HMMThreeTriangularFully --seed 0 --hidden_size 50 --input_dim 1 --gpu 1
"""

import argparse
import sys
import torch
import random
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.manager import Manager
import scripts.config

def set_random_seed(seed):
    print(f"Setting random seed to: {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train RNN model with custom parameters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Name of the base configuration to use (e.g., HMMThreeTriangularFully)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")
    parser.add_argument("--hidden_size", type=int, default=None, help="Number of hidden units in the RNN (overrides config default)")
    parser.add_argument("--input_dim", type=int, default=None, help="Input dimension for the RNN (overrides config default)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose training output")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config default)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for training (overrides config default)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training (overrides config default)")
    return parser.parse_args()


def load_and_modify_config(config_name, args):
    if not hasattr(scripts.config, config_name):
        available_configs = [name for name in dir(scripts.config) if not name.startswith('_') and name[0].isupper()]
        print(f"Error: Configuration '{config_name}' not found.")
        print(f"Available configurations: {', '.join(available_configs)}")
        sys.exit(1)
    
    Config = getattr(scripts.config, config_name)
    config = Config.get_config()
    
    print(f"\nBase configuration loaded: {config_name}")
    print(f"Original settings:")
    print(f"  - Hidden size: {config['hidden_size']}")
    print(f"  - Input size: {config['input_size']}")
    print(f"  - Epochs: {config['epochs']}")
    print(f"  - Learning rates: {config['learning_rates']}")
    print(f"  - Batch size: {config['batch_size']}")
    
    if args.hidden_size is not None:
        config['hidden_size'] = args.hidden_size
        print(f"\n✓ Overriding hidden_size: {args.hidden_size}")
    
    if args.input_dim is not None:
        config['input_size'] = args.input_dim
        print(f"✓ Overriding input_size: {args.input_dim}")
    
    if args.epochs is not None:
        config['epochs'] = args.epochs
        print(f"✓ Overriding epochs: {args.epochs}")
    
    if args.learning_rate is not None:
        config['learning_rates'] = [args.learning_rate]
        print(f"✓ Overriding learning_rate: {args.learning_rate}")
    
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        print(f"✓ Overriding batch_size: {args.batch_size}")
    
    return config


def main():
    args = parse_arguments()
    
    print("="*70)
    print("RNN Training with Custom Parameters")
    print("="*70)
    
    set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"✓ Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        print("✓ CUDA not available, using CPU")

    config = load_and_modify_config(args.config, args)
    custom_config_name = f"{args.config}_seed{args.seed}"
    if args.hidden_size is not None:
        custom_config_name += f"_h{args.hidden_size}"
    if args.input_dim is not None:
        custom_config_name += f"_i{args.input_dim}"
    print(f"\nCustom configuration name: {custom_config_name}")
    print("\nInitializing Manager...")
    manager = Manager(config_dict=config, config_name=custom_config_name)

    print("\n" + "="*70)
    print("Starting Experiment")
    print("="*70 + "\n")
    
    try:
        results = manager.run_experiment(verbose=args.verbose)
        
        print("\n" + "="*70)
        print("Experiment Completed Successfully!")
        print("="*70)
        print(f"\nResults saved in: {manager.config_dir}")
        print(f"Best loss: {results['best_loss']:.6f}")
        print(f"Duration: {results['experiment_duration_minutes']:.2f} minutes")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("Error during experiment:")
        print("="*70)
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

import os
import torch
import pickle
import geomloss
import numpy as np
import json
from datetime import datetime
from pathlib import Path

class ExperimentManager:
    """
    Class for managing experiment execution, data handling, and result storage.
    """
    def __init__(self, config):
        """
        Initialize the experiment manager.
        
        Args:
            config (dict): Experiment configuration
        """
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """Set up the directory structure for the experiment"""
        # Create base folder
        self.base_folder = self.config.get("experiment_name", "experiment")
        
        # Support for creating experiment inside a parent directory
        self.base_path = Path(self.base_folder)
        self.base_path.mkdir(exist_ok=True, parents=True)
        
        # Create subfolders
        self.models_path = self.base_path / "models"
        self.figs_path = self.base_path / "figs"
        self.data_path = self.base_path / "data"
        
        self.models_path.mkdir(exist_ok=True)
        self.figs_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
        print(f"Created directory structure in {self.base_path}")
        
    def save_config(self):
        """Save the configuration to a file"""
        # Save as pickle for program use
        config_path = self.base_path / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(self.config, f)
            
        # Also save as JSON for human readability
        json_path = self.base_path / "config.json"
        with open(json_path, "w") as f:
            # Convert numpy arrays to lists if present
            config_json = {}
            for key, value in self.config.items():
                if isinstance(value, np.ndarray):
                    config_json[key] = value.tolist()
                else:
                    config_json[key] = value
            json.dump(config_json, f, indent=4)
            
        print(f"Configuration saved to {config_path} and {json_path}")
        
    def load_config(self, path):
        """
        Load a configuration from a file.
        
        Args:
            path (str): Path to the configuration file
        """
        path = Path(path)
        if path.suffix == '.json':
            with open(path, "r") as f:
                self.config = json.load(f)
        else:
            with open(path, "rb") as f:
                self.config = pickle.load(f)
        print(f"Configuration loaded from {path}")
        
    def run_experiment(self, hmm_generator, rnn_model, model_tester, pca_analyzer):
        """
        Run the complete experiment pipeline.
        
        Args:
            hmm_generator: HMMGenerator instance
            rnn_model: RNNModel instance
            model_tester: ModelTester instance
            pca_analyzer: PCAAnalyzer instance
            
        Returns:
            dict: Dictionary with experiment results
        """
        # Record start time
        start_time = datetime.now()
        print(f"Starting experiment at {start_time}")
        
        # 1. Generate HMM data
        print("Generating HMM data...")
        one_hot_sequences, sampled_states = hmm_generator.generate_sequences(
            self.config["num_seq"], self.config["seq_len"]
        )
        
        # 2. Split the data
        print("Splitting data...")
        data_splits = hmm_generator.split_data(one_hot_sequences, sampled_states)
        
        # Save the data
        data_path = self.data_path / "hmm_sequences.pkl"
        with open(data_path, "wb") as f:
            # Convert PyTorch tensors to NumPy arrays for saving
            data_to_save = {
                key: value.numpy() if isinstance(value, torch.Tensor) else value
                for key, value in data_splits.items()
            }
            pickle.dump(data_to_save, f)
        print(f"HMM data saved to {data_path}")
        
        # 3. Train the RNN
        print("Training RNN model...")
        train_seq = data_splits["train_seq"]
        val_seq = data_splits["val_seq"]
        
        # Setup Sinkhorn loss for training
        criterion = geomloss.SamplesLoss(blur=0.3)
        
        # Try different learning rates
        best_loss = float('inf')
        best_lr = None
        
        for lr in self.config.get("learning_rates", [0.001]):
            print(f"Training with learning rate: {lr}")
            rnn_model.train_model(
                train_seq=train_seq,
                val_seq=val_seq,
                batch_size=self.config.get("batch_size", 4096),
                lr=lr,
                tau=self.config.get("tau", 1.0),
                epochs=self.config.get("epochs", 1000),
                grad_clip=self.config.get("grad_clip", 0.9),
                init=self.config.get("init", True),
                criterion=criterion
            )
            
            # Check if this is the best model
            if rnn_model.best_loss < best_loss:
                best_loss = rnn_model.best_loss
                best_lr = lr
                
                # Save the model
                model_filename = f"{hmm_generator.states}HMM_{hmm_generator.outputs}Outputs_{self.config['emission_method']}_{self.config['num_seq']//1000}kData_{lr}lr_{best_loss:.1f}Loss.pth"
                model_path = self.models_path / model_filename
                rnn_model.save_model(model_path)
                
                # Plot losses
                loss_plot_path = self.figs_path / "loss_curves.pdf"
                rnn_model.plot_losses(loss_plot_path)
        
        print(f"Best model achieved with learning rate {best_lr} and loss {best_loss:.6f}")
        
        # 4. Run tests
        print("Running model tests...")
        test_results = model_tester.run_all_tests()
        model_tester.generate_plots(test_results, save_path=self.figs_path)
        
        # 5. Run PCA analysis
        print("Running PCA analysis...")
        pca_results = pca_analyzer.run_analysis(
            dynamics_mode="full", 
            save_path=self.figs_path
        )
        
        # Record end time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        
        # 6. Save experiment results
        results = {
            "config": self.config,
            "best_loss": best_loss,
            "best_lr": best_lr,
            "test_results": test_results,
            "pca_results": pca_results,
            "experiment_duration_minutes": duration,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_path = self.base_path / "results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
            
        # Also save a summary as JSON
        summary = {
            "config": {k: str(v) for k, v in self.config.items()},
            "best_loss": float(best_loss),
            "best_lr": float(best_lr),
            "experiment_duration_minutes": float(duration),
            "start_time": results["start_time"],
            "end_time": results["end_time"],
            "explained_variance": pca_results["explained_variance"].tolist() if isinstance(pca_results["explained_variance"], np.ndarray) else pca_results["explained_variance"]
        }
        
        summary_path = self.base_path / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Experiment results saved to {results_path}")
        print(f"Experiment summary saved to {summary_path}")
        print(f"Experiment completed in {duration:.2f} minutes")
        
        return results

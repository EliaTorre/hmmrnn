import os
import torch
import pickle
import geomloss
import numpy as np
import json
import math
import datetime
from pathlib import Path
import importlib
import random
from collections import defaultdict

from scripts.hmm import HMM
from scripts.rnn import RNN
from scripts.test import Test
from scripts.reverse import Reverse
import scripts.config
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch
from fixed_point_finder.plot_utils import plot_fps

class Manager:
    """
    Class for managing experiment execution, data handling, and result storage.
    """
    def __init__(self, config_dict=None, config_name=None):
        """
        Initialize the experiment manager.
        
        Args:
            config (dict, optional): Experiment configuration dict
            config_name (str, optional): Name of a configuration class in CONFIG module
        """
        if config_name and not config_dict:
            # Load config from CONFIG module
            if hasattr(scripts.config, config_name):
                Config = getattr(scripts.config, config_name)
                self.config = Config.get_config()
                self.config_name = config_name
            else:
                raise ValueError(f"Configuration '{config_name}' not found in config module")
        else:
            self.config = config_dict if config_dict else scripts.config.DefaultConfig.get_config()
            self.config_name = config_name if config_name else "CustomConfig"
        
        # Set timestamp for this experiment
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.setup_dir()
        
    def setup_dir(self):
        """Set up the directory structure for the experiment"""
        # Create base Experiments folder
        experiments_dir = Path("Experiments")
        experiments_dir.mkdir(exist_ok=True)
        
        # Create timestamped folder for this experiment
        self.experiment_dir = experiments_dir / self.timestamp
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create config-specific folder
        self.config_dir = self.experiment_dir / self.config_name
        self.config_dir.mkdir(exist_ok=True)
        
        # Create subfolders
        self.models_path = self.config_dir / "models"
        self.figs_path = self.config_dir / "figs"
        self.data_path = self.config_dir / "data"
        
        self.models_path.mkdir(exist_ok=True)
        self.figs_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
        print(f"Created directory structure in {self.config_dir}")
        
    def save_config(self):
        """Save the configuration to a file"""
        # Save as pickle for program use
        config_path = self.config_dir / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(self.config, f)
            
        # Also save as JSON for human readability
        json_path = self.config_dir / "config.json"
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
    
    def clear_gpu_memory(self):
        """
        Clear GPU memory to prevent memory leaks between experiments.
        """
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    
    def run_training(self, verbose=False):
        """
        Train an RNN model on HMM data.
        
        Args:
            verbose (bool): Whether to print detailed progress
            
        Returns:
            tuple: (hmm, rnn, data_splits)
        """
        print("Starting RNN training...")
        
        # Create HMM and RNN models
        hmm = HMM(
            states=self.config["states"],
            outputs=self.config["outputs"],
            stay_prob=self.config["stay_prob"],
            target_prob=self.config.get("target_prob", 0.05),
            transition_method=self.config.get("transition_method", "target_prob"),
            emission_method=self.config["emission_method"],
            custom_transition_matrix=self.config.get("custom_transition_matrix", None),
            custom_emission_matrix=self.config.get("custom_emission_matrix", None)
        )
        
        rnn = RNN(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            output_size=self.config["outputs"],
            biased=self.config["biased"]
        )
        
        # Generate HMM data
        print("Generating HMM data...")
        one_hot_sequences, sampled_states = hmm.gen_seq(
            self.config["num_seq"], self.config["seq_len"]
        )
        
        # Split the data
        print("Splitting data...")
        data_splits = hmm.split_data(one_hot_sequences, sampled_states)
        
        # Save the data
        #data_path = self.data_path / "hmm_sequences.pkl"
        #with open(data_path, "wb") as f:
            # Convert PyTorch tensors to NumPy arrays for saving
            #data_to_save = {
                #key: value.numpy() if isinstance(value, torch.Tensor) else value
                #for key, value in data_splits.items()
            #}
            #pickle.dump(data_to_save, f)
        #print(f"HMM data saved to {data_path}")
        
        # Setup Sinkhorn loss for training
        criterion = geomloss.SamplesLoss(blur=0.3)
        
        # Try different learning rates
        best_loss = float('inf')
        best_lr = None
        
        for lr in self.config.get("learning_rates", [0.001]):
            print(f"Training with learning rate: {lr}")
            rnn.train_model(
                train_seq=data_splits["train_seq"],
                val_seq=data_splits["val_seq"],
                batch_size=self.config.get("batch_size", 4096),
                lr=lr,
                tau=self.config.get("tau", 1.0),
                epochs=self.config.get("epochs", 1000),
                grad_clip=self.config.get("grad_clip", 0.9),
                init=self.config.get("init", True),
                criterion=criterion,
                verbose=verbose
            )
            
            # Check if this is the best model
            if rnn.best_loss < best_loss:
                best_loss = rnn.best_loss
                best_lr = lr
                
                # Save the model
                model_filename = f"{hmm.states}HMM_{hmm.outputs}Outputs_{self.config['emission_method']}_{self.config['num_seq']//1000}kData_{lr}lr_{best_loss:.1f}Loss.pth"
                model_path = self.models_path / model_filename
                rnn.save_model(model_path)
                
                # Plot losses
                loss_plot_path = self.figs_path / "loss_curves.pdf"
                rnn.plot_losses(loss_plot_path)
        
        print(f"Best model achieved with learning rate {best_lr} and loss {best_loss:.6f}")
        
        # Save training results
        training_results = {
            "best_loss": best_loss,
            "best_lr": best_lr,
            "training_losses": rnn.train_losses,
            "validation_losses": rnn.val_losses
        }
        
        results_path = self.config_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(training_results, f, indent=4)
        
        # Clear GPU memory
        self.clear_gpu_memory()
        
        return hmm, rnn, data_splits
    
    def run_tests(self, hmm=None, rnn=None):
        """
        Run tests on the trained model.
        
        Args:
            hmm (HMM, optional): HMM model
            rnn (RNN, optional): RNN model
            
        Returns:
            dict: Test results
        """
        print("Running model tests...")
        
        # If models are not provided, create them
        if hmm is None or rnn is None:
            # Load the most recent model
            model_files = list(self.models_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Run training first.")
            
            latest_model = max(model_files, key=os.path.getmtime)
            
            hmm = HMM(
                states=self.config["states"],
                outputs=self.config["outputs"],
                stay_prob=self.config["stay_prob"],
                target_prob=self.config.get("target_prob", 0.05),
                transition_method=self.config.get("transition_method", "target_prob"),
                emission_method=self.config["emission_method"]
            )
            
            rnn = RNN(
                input_size=self.config["input_size"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                output_size=self.config["outputs"],
                biased=self.config["biased"]
            )
            
            rnn.load_model(latest_model)
        
        # Create model tester
        model_tester = Test(
            hmm=hmm,
            rnn=rnn,
            num_seq=self.config["num_seq"],
            seq_len=self.config["seq_len"],
            outputs=self.config["outputs"]
        )
        
        # Run tests
        test_results = model_tester.run_all()
        
        # Generate plots with model info
        model_info = {
            "states": self.config["states"],
            "hidden_size": self.config["hidden_size"],
            "input_size": self.config["input_size"]
        }
        model_tester.gen_plots(test_results, save_path=self.figs_path, model_info=model_info)
        
        # Save test results
        #test_results_path = self.config_dir / "test_results.pkl"
        #with open(test_results_path, "wb") as f:
            #pickle.dump(test_results, f)
        
        # Clear GPU memory
        self.clear_gpu_memory() 
        
        return test_results
    
    def run_reverse(self, rnn=None):
        """
        Run reverse-engineering analysis on the trained model.
        
        Args:
            rnn (RNN, optional): RNN model
            
        Returns:
            dict: Analysis results
        """
        print("Running reverse-engineering analysis...")
        
        # If rnn is not provided, load it
        if rnn is None:
            # Load the most recent model
            model_files = list(self.models_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Run training first.")
            
            latest_model = max(model_files, key=os.path.getmtime)
            
            rnn = RNN(
                input_size=self.config["input_size"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                output_size=self.config["outputs"],
                biased=self.config["biased"]
            )
            
            rnn.load_model(latest_model)
        
        # Create REVERSE analyzer
        pca_analyzer = Reverse(
            rnn=rnn,
            num_seq=self.config["num_seq"],
            seq_len=self.config["seq_len"],
            outputs=self.config["outputs"]
        )
        
        # Create model info dictionary
        model_info = {
            "states": self.config["states"],
            "hidden_size": self.config["hidden_size"],
            "input_size": self.config["input_size"]
        }
        
        # Run PCA analysis
        pca_results = pca_analyzer.run_analysis(
            dynamics_mode="full", 
            save_path=self.figs_path,
            model_info=model_info
        )
        
        # Save PCA results
        #pca_results_path = self.config_dir / "pca_results.pkl"
        #with open(pca_results_path, "wb") as f:
            #pickle.dump(pca_results, f)
        
        # Clear GPU memory
        self.clear_gpu_memory()
        
        return pca_results
    
    def run_experiment(self, verbose=False):
        """
        Run the complete experiment pipeline.
        
        Args:
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Dictionary with experiment results
        """
        # Record start time
        start_time = datetime.datetime.now()
        print(f"Starting experiment at {start_time}")
        
        # Save configuration
        self.save_config()
        
        # 1. Train RNN model
        hmm, rnn, data_splits = self.run_training(verbose=verbose)
        
        # 2. Run tests
        test_results = self.run_tests(hmm, rnn)
        
        # 3. Run PCA analysis
        pca_results = self.run_reverse(rnn)
        
        # Record end time
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        
        # 4. Save experiment results
        results = {
            "config": self.config,
            "best_loss": rnn.best_loss,
            "best_lr": None,  # Will be filled below
            "test_results": test_results,
            "pca_results": pca_results,
            "experiment_duration_minutes": duration,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Find the best learning rate
        for lr in self.config.get("learning_rates", [0.001]):
            model_file = list(self.models_path.glob(f"*{lr}lr_*.pth"))
            if model_file:
                results["best_lr"] = lr
                break
        
        #results_path = self.config_dir / "results.pkl"
        #with open(results_path, "wb") as f:
            #pickle.dump(results, f)
            
        # Also save a summary as JSON
        summary = {
            "config_name": self.config_name,
            "hmm": {
                "states": self.config["states"],
                "outputs": self.config["outputs"],
                "emission_method": self.config["emission_method"]
            },
            "rnn": {
                "hidden_size": self.config["hidden_size"]
            },
            "best_loss": float(rnn.best_loss),
            "best_lr": results["best_lr"],
            "experiment_duration_minutes": float(duration),
            "start_time": results["start_time"],
            "end_time": results["end_time"],
            "explained_variance": pca_results["explained_variance"].tolist() if isinstance(pca_results["explained_variance"], np.ndarray) else pca_results["explained_variance"]
        }
        
        summary_path = self.config_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        print(f"Experiment results saved to {results_path}")
        print(f"Experiment summary saved to {summary_path}")
        print(f"Experiment completed in {duration:.2f} minutes")
        
        return results
    
    def run_multiple_experiments(self, config_names, verbose=False):
        """
        Run multiple experiments with different configurations.
        
        Args:
            config_names (list): List of configuration class names from CONFIG module
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Dictionary with summary of all experiments
        """
        # Save the current configuration and timestamp
        original_config = self.config
        original_config_name = self.config_name
        original_timestamp = self.timestamp
        
        # Initialize results dictionary
        all_results = {
            "timestamp": self.timestamp,
            "experiments": []
        }
        
        # Run each experiment
        for config_name in config_names:
            print(f"="*60)
            print(f"Running experiment with config: {config_name}")
            print(f"="*60)
            
            # Check if the config exists
            if not hasattr(scripts.config, config_name):
                print(f"Configuration '{config_name}' not found in CONFIG module. Skipping.")
                experiment_result = {
                    "config_name": config_name,
                    "success": False,
                    "error": "Configuration not found"
                }
                all_results["experiments"].append(experiment_result)
                continue
            
            # Create a new manager for this configuration
            # We want to keep the same timestamp for all experiments in this batch
            self.config_name = config_name
            Config = getattr(scripts.config, config_name)
            self.config = Config.get_config()
            
            # Create the config directory
            self.config_dir = self.experiment_dir / config_name
            self.config_dir.mkdir(exist_ok=True)
            
            # Create subfolders
            self.models_path = self.config_dir / "models"
            self.figs_path = self.config_dir / "figs"
            self.data_path = self.config_dir / "data"
            
            self.models_path.mkdir(exist_ok=True)
            self.figs_path.mkdir(exist_ok=True)
            self.data_path.mkdir(exist_ok=True)
            
            print(f"Created directory structure in {self.config_dir}")
            
            # Run the experiment
            try:
                results = self.run_experiment(verbose=verbose)
                experiment_result = {
                    "config_name": config_name,
                    "success": True,
                    "best_loss": float(results["best_loss"]),
                    "experiment_duration_minutes": float(results["experiment_duration_minutes"])
                }
            except Exception as e:
                import traceback
                print(f"Error running experiment: {str(e)}")
                print(traceback.format_exc())
                experiment_result = {
                    "config_name": config_name,
                    "success": False,
                    "error": str(e)
                }
            
            all_results["experiments"].append(experiment_result)
        
        # Restore original configuration
        self.config = original_config
        self.config_name = original_config_name
        self.timestamp = original_timestamp
        
        # Save overall results
        batch_results_path = self.experiment_dir / "batch_results.json"
        with open(batch_results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"="*60)
        print(f"All experiments completed.")
        print(f"Results saved in: {self.experiment_dir}")
        print(f"="*60)
        
        return all_results

    def run_training_evo(self, verbose=False):
        """
        Train an RNN model on HMM data, saving intermediate models every 5 epochs (including after the first epoch).
        Tests and plots are generated only for the final model. Intermediate models are stored in a directory
        with the timestamp. Returns a results dictionary.
        
        Args:
            verbose (bool): Whether to print detailed progress
            
        Returns:
            dict: Dictionary with experiment results
        """
        # Record start time
        start_time = datetime.datetime.now()
        print(f"Starting RNN training with evolution saving at {start_time}")

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed(0)
        
        # Use the timestamp directly without adding '_evo'
        experiments_dir = Path("Experiments")
        self.experiment_dir = experiments_dir / self.timestamp
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Create config-specific folder
        self.config_dir = self.experiment_dir / self.config_name
        self.config_dir.mkdir(exist_ok=True)
        
        # Create subfolders
        self.models_path = self.config_dir / "models"
        self.figs_path = self.config_dir / "figs"
        self.data_path = self.config_dir / "data"
        self.models_path.mkdir(exist_ok=True)
        self.figs_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        
        # Create evolution models directory
        evolution_models_path = self.models_path / "evolution"
        evolution_models_path.mkdir(exist_ok=True)
        
        # Create HMM and RNN models
        hmm = HMM(
            states=self.config["states"],
            outputs=self.config["outputs"],
            stay_prob=self.config["stay_prob"],
            target_prob=self.config.get("target_prob", 0.05),
            transition_method=self.config.get("transition_method", "target_prob"),
            emission_method=self.config["emission_method"],
            custom_transition_matrix=self.config.get("custom_transition_matrix", None),
            custom_emission_matrix=self.config.get("custom_emission_matrix", None)
        )
        
        rnn = RNN(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            output_size=self.config["outputs"],
            biased=self.config["biased"]
        )
        
        # Generate HMM data
        print("Generating HMM data...")
        one_hot_sequences, sampled_states = hmm.gen_seq(
            self.config["num_seq"], self.config["seq_len"]
        )
        
        # Split the data
        print("Splitting data...")
        data_splits = hmm.split_data(one_hot_sequences, sampled_states)
        
        # Save the data
        #data_path = self.data_path / "hmm_sequences.pkl"
        #with open(data_path, "wb") as f:
            #data_to_save = {
                #key: value.numpy() if isinstance(value, torch.Tensor) else value
                #for key, value in data_splits.items()
            #}
            #pickle.dump(data_to_save, f)
        #print(f"HMM data saved to {data_path}")
        
        # Setup Sinkhorn loss
        criterion = geomloss.SamplesLoss(blur=0.3)
        
        # Use the first learning rate for simplicity
        lr = self.config.get("learning_rates", [0.001])[0]
        print(f"Training with learning rate: {lr}")
        
        # Train the model with saving every 5 epochs
        rnn.train_model(
            train_seq=data_splits["train_seq"],
            val_seq=data_splits["val_seq"],
            batch_size=self.config.get("batch_size", 4096),
            lr=lr,
            tau=self.config.get("tau", 1.0),
            epochs=self.config.get("epochs", 1000),
            grad_clip=self.config.get("grad_clip", 0.9),
            init=self.config.get("init", True),
            criterion=criterion,
            verbose=verbose,
            save_interval=1,
            save_path=evolution_models_path
        )
        
        # Save the final model
        final_model_filename = f"{hmm.states}HMM_{hmm.outputs}Outputs_{self.config['emission_method']}_{self.config['num_seq']//1000}kData_{lr}lr_{rnn.best_loss:.1f}Loss.pth"
        final_model_path = self.models_path / final_model_filename
        torch.save(rnn.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Plot losses with additional information in the title
        loss_plot_path = self.figs_path / "loss_curves.pdf"
        title_prefix = f"Loss Curves - States: {self.config['states']}, Hidden: {self.config['hidden_size']}, Input: {self.config['input_size']}"
        rnn.plot_losses(loss_plot_path, title_prefix=title_prefix)
        
        # Generate model evolution plot
        try:
            from scripts.mechint_evo import model_evolution
            import matplotlib.pyplot as plt
            
            # Get the path to the evolution models directory
            evolution_models_path = self.models_path / "evolution"
            
            # Get the path to the best model
            best_model_path = final_model_path
            
            # Create the model evolution plot
            evolution_plot_path = self.figs_path / "model_evolution.pdf"
            
            # Call model_evolution function with title prefix and model info
            title_prefix = f"Model Evolution - States: {self.config['states']}, Hidden: {self.config['hidden_size']}, Input: {self.config['input_size']}"
            fig = model_evolution(
                evolution_dir=evolution_models_path,
                best_model_path=best_model_path,
                num_steps_best=30000,
                num_steps_other=50,
                best_steps_to_plot=1000,
                title_prefix=title_prefix,
                input_size=self.config["input_size"],
                hidden_size=self.config["hidden_size"],
                num_states=self.config["states"]
            )
            
            # Save the figure
            fig.savefig(evolution_plot_path)
            plt.close(fig)
            
            print(f"Model evolution plot saved to {evolution_plot_path}")
        except Exception as e:
            print(f"Error generating model evolution plot: {str(e)}")
        
        # Run tests and reverse analysis for the final model
        test_results = self.run_tests(hmm, rnn)
        pca_results = self.run_reverse(rnn)
        
        # Record end time
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        
        # Construct results dictionary
        results = {
            "config": self.config,
            "best_loss": float(rnn.best_loss),
            "learning_rate": lr,
            "train_losses": rnn.train_losses,
            "val_losses": rnn.val_losses,
            "test_results": test_results,
            "pca_results": pca_results,
            "experiment_duration_minutes": float(duration),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save results
        #results_path = self.config_dir / "results.pkl"
        #with open(results_path, "wb") as f:
            #pickle.dump(results, f)
        
        # Save a summary as JSON
        summary = {
            "config_name": self.config_name,
            "hmm": {
                "states": self.config["states"],
                "outputs": self.config["outputs"],
                "emission_method": self.config["emission_method"]
            },
            "rnn": {
                "hidden_size": self.config["hidden_size"]
            },
            "best_loss": float(rnn.best_loss),
            "learning_rate": lr,
            "experiment_duration_minutes": float(duration),
            "start_time": results["start_time"],
            "end_time": results["end_time"],
            "explained_variance": pca_results["explained_variance"].tolist() if isinstance(pca_results["explained_variance"], np.ndarray) else pca_results["explained_variance"]
        }
        
        summary_path = self.config_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        print(f"Experiment results saved to {results_path}")
        print(f"Experiment summary saved to {summary_path}")
        print(f"Experiment completed in {duration:.2f} minutes")
        
        # Clear GPU memory
        self.clear_gpu_memory()
        
        return results

    def run_grid_search(self, configs, hidden_sizes, input_sizes, seeds=[0], verbose=False):
        """
        Run a grid search over multiple configurations, RNN hidden unit sizes, and input sizes.
        
        Args:
            configs (list): List of configuration names from CONFIG module
            hidden_sizes (list): List of values for the number of RNN hidden units
            input_sizes (list): List of values for the input size
            verbose (bool): Whether to print detailed progress
            seeds (list): List of random seeds to use for each run (default: [0])
            
        Returns:
            dict: Dictionary with summary of all experiments
        """
        # Create a main timestamp for this grid search
        grid_search_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting grid search at {grid_search_timestamp}")
        
        # Create the main grid search directory
        experiments_dir = Path("Experiments")
        experiments_dir.mkdir(exist_ok=True)
        grid_search_dir = experiments_dir / f"grid_search_{grid_search_timestamp}"
        grid_search_dir.mkdir(exist_ok=True)
        
        # Initialize results dictionary
        all_results = {
            "timestamp": grid_search_timestamp,
            "configs": configs,
            "hidden_sizes": hidden_sizes,
            "input_sizes": input_sizes,
            "seeds": seeds,
            "experiments": []
        }
        
        # Save the original configuration and timestamp
        original_config = self.config.copy() if self.config else None
        original_config_name = self.config_name
        original_timestamp = self.timestamp
        
        # Loop through all combinations
        for config_name in configs:
            print(f"="*60)
            print(f"Processing configuration: {config_name}")
            
            # Check if the config exists
            if not hasattr(scripts.config, config_name):
                print(f"Configuration '{config_name}' not found in CONFIG module. Skipping.")
                continue
            
            # Create config directory
            config_dir = grid_search_dir / config_name
            config_dir.mkdir(exist_ok=True)
            
            # Get the config
            Config = getattr(scripts.config, config_name)
            base_config = Config.get_config()
            
            for hidden_size in hidden_sizes:
                print(f"-"*40)
                print(f"Processing hidden size: {hidden_size}")
                
                # Create hidden size directory
                hidden_dir = config_dir / f"hidden_{hidden_size}"
                hidden_dir.mkdir(exist_ok=True)
                
                for input_size in input_sizes:
                    print(f"Processing input size: {input_size}")
                    
                    # Create input size directory
                    input_dir = hidden_dir / f"input_{input_size}"
                    input_dir.mkdir(exist_ok=True)
                    
                    for seed in seeds:
                        print(f"Processing seed: {seed}")
                        
                        # Create seed directory
                        seed_dir = input_dir / f"seed_{seed}"
                        seed_dir.mkdir(exist_ok=True)
                        
                        # Create a modified config with the current hidden_size and input_size
                        modified_config = base_config.copy()
                        modified_config["hidden_size"] = hidden_size
                        modified_config["input_size"] = input_size
                        
                        # Create a new manager for this specific configuration
                        experiment_manager = Manager(config_dict=modified_config, config_name=f"{config_name}_h{hidden_size}_i{input_size}_s{seed}")
                        
                        # Save the original directory structure created by the Manager
                        original_experiment_dir = experiment_manager.experiment_dir
                        original_config_dir = experiment_manager.config_dir
                        
                        # Set up the directory structure to match our grid search structure
                        experiment_manager.experiment_dir = seed_dir
                        experiment_manager.config_dir = seed_dir
                        experiment_manager.models_path = seed_dir / "models"
                        experiment_manager.figs_path = seed_dir / "figs"
                        experiment_manager.data_path = seed_dir / "data"
                        
                        # Create the necessary directories
                        experiment_manager.models_path.mkdir(exist_ok=True)
                        experiment_manager.figs_path.mkdir(exist_ok=True)
                        experiment_manager.data_path.mkdir(exist_ok=True)
                        
                        # Override the run_training_evo method to use our directory structure and seed
                        def custom_run_training_evo(self, verbose=False):
                            # Record start time
                            start_time = datetime.datetime.now()
                            print(f"Starting RNN training with evolution saving at {start_time}")
                            
                            # Set random seeds
                            torch.manual_seed(seed)
                            random.seed(seed)
                            np.random.seed(seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed(seed)
                            
                            # Create evolution models directory
                            evolution_models_path = self.models_path / "evolution"
                            evolution_models_path.mkdir(exist_ok=True)
                            
                            # Create HMM and RNN models
                            hmm = HMM(
                                states=self.config["states"],
                                outputs=self.config["outputs"],
                                stay_prob=self.config["stay_prob"],
                                target_prob=self.config.get("target_prob", 0.05),
                                transition_method=self.config.get("transition_method", "target_prob"),
                                emission_method=self.config["emission_method"],
                                custom_transition_matrix=self.config.get("custom_transition_matrix", None),
                                custom_emission_matrix=self.config.get("custom_emission_matrix", None)
                            )
                            
                            rnn = RNN(
                                input_size=self.config["input_size"],
                                hidden_size=self.config["hidden_size"],
                                num_layers=self.config["num_layers"],
                                output_size=self.config["outputs"],
                                biased=self.config["biased"]
                            )
                            
                            # Generate HMM data
                            print("Generating HMM data...")
                            one_hot_sequences, sampled_states = hmm.gen_seq(
                                self.config["num_seq"], self.config["seq_len"]
                            )
                            
                            # Split the data
                            print("Splitting data...")
                            data_splits = hmm.split_data(one_hot_sequences, sampled_states)
                            
                            # Save the data
                            #data_path = self.data_path / "hmm_sequences.pkl"
                            #with open(data_path, "wb") as f:
                                #data_to_save = {
                                    #key: value.numpy() if isinstance(value, torch.Tensor) else value
                                    #for key, value in data_splits.items()
                                #}
                                #pickle.dump(data_to_save, f)
                            #print(f"HMM data saved to {data_path}")
                            
                            # Setup Sinkhorn loss
                            criterion = geomloss.SamplesLoss(blur=0.3)
                            
                            # Use the first learning rate for simplicity
                            lr = self.config.get("learning_rates", [0.001])[0]
                            print(f"Training with learning rate: {lr}")
                            
                            # Train the model with saving every epoch
                            rnn.train_model(
                                train_seq=data_splits["train_seq"],
                                val_seq=data_splits["val_seq"],
                                batch_size=self.config.get("batch_size", 4096),
                                lr=lr,
                                tau=self.config.get("tau", 1.0),
                                epochs=self.config.get("epochs", 1000),
                                grad_clip=self.config.get("grad_clip", 0.9),
                                init=self.config.get("init", True),
                                criterion=criterion,
                                verbose=verbose,
                                save_interval=1,
                                save_path=evolution_models_path
                            )
                            
                            # Save the final model
                            final_model_filename = f"{hmm.states}HMM_{hmm.outputs}Outputs_{self.config['emission_method']}_{self.config['num_seq']//1000}kData_{lr}lr_{rnn.best_loss:.1f}Loss.pth"
                            final_model_path = self.models_path / final_model_filename
                            torch.save(rnn.state_dict(), final_model_path)
                            print(f"Final model saved to {final_model_path}")
                            
                            # Plot losses with additional information in the title
                            loss_plot_path = self.figs_path / "loss_curves.pdf"
                            title_prefix = f"Loss Curves - States: {self.config['states']}, Hidden: {self.config['hidden_size']}, Input: {self.config['input_size']}, Seed: {seed}"
                            rnn.plot_losses(loss_plot_path, title_prefix=title_prefix)
                            
                            # Generate model evolution plot
                            try:
                                from scripts.mechint_evo import model_evolution
                                import matplotlib.pyplot as plt
                                
                                # Get the path to the evolution models directory
                                evolution_models_path = self.models_path / "evolution"
                                
                                # Get the path to the best model
                                best_model_path = final_model_path
                                
                                # Create the model evolution plot
                                evolution_plot_path = self.figs_path / "model_evolution.pdf"
                                
                                # Call model_evolution function with title prefix and model info
                                title_prefix = f"Model Evolution - States: {self.config['states']}, Hidden: {self.config['hidden_size']}, Input: {self.config['input_size']}, Seed: {seed}"
                                fig = model_evolution(
                                    evolution_dir=evolution_models_path,
                                    best_model_path=best_model_path,
                                    num_steps_best=30000,
                                    num_steps_other=50,
                                    best_steps_to_plot=1000,
                                    title_prefix=title_prefix,
                                    input_size=self.config["input_size"],
                                    hidden_size=self.config["hidden_size"],
                                    num_states=self.config["states"]
                                )
                                
                                # Save the figure
                                fig.savefig(evolution_plot_path)
                                plt.close(fig)
                                
                                print(f"Model evolution plot saved to {evolution_plot_path}")
                            except Exception as e:
                                import traceback
                                print(f"Error generating model evolution plot: {str(e)}")
                                print(traceback.format_exc())
                            
                            # Run tests and reverse analysis for the final model
                            #test_results = self.run_tests(hmm, rnn)
                            pca_results = self.run_reverse(rnn)
                            
                            # Record end time
                            end_time = datetime.datetime.now()
                            duration = (end_time - start_time).total_seconds() / 60  # in minutes
                            
                            # Construct results dictionary
                            results = {
                                "config": self.config,
                                "best_loss": float(rnn.best_loss),
                                "learning_rate": lr,
                                "train_losses": rnn.train_losses,
                                "val_losses": rnn.val_losses,
                                #"test_results": test_results,
                                "pca_results": pca_results,
                                "experiment_duration_minutes": float(duration),
                                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "seed": seed
                            }
                            
                            # Save results
                            #results_path = self.config_dir / "results.pkl"
                            #with open(results_path, "wb") as f:
                                #pickle.dump(results, f)
                            
                            # Save a summary as JSON
                            summary = {
                                "config_name": self.config_name,
                                "hmm": {
                                    "states": self.config["states"],
                                    "outputs": self.config["outputs"],
                                    "emission_method": self.config["emission_method"]
                                },
                                "rnn": {
                                    "hidden_size": self.config["hidden_size"],
                                    "input_size": self.config["input_size"]
                                },
                                "seed": seed,
                                "best_loss": float(rnn.best_loss),
                                "learning_rate": lr,
                                "experiment_duration_minutes": float(duration),
                                "start_time": results["start_time"],
                                "end_time": results["end_time"],
                                "explained_variance": pca_results["explained_variance"].tolist() if isinstance(pca_results["explained_variance"], np.ndarray) else pca_results["explained_variance"]
                            }
                            
                            summary_path = self.config_dir / "summary.json"
                            with open(summary_path, "w") as f:
                                json.dump(summary, f, indent=4)
                            
                            print(f"Experiment results saved to {results_path}")
                            print(f"Experiment summary saved to {summary_path}")
                            print(f"Experiment completed in {duration:.2f} minutes")
                            
                            # Clear GPU memory
                            self.clear_gpu_memory()
                            
                            return results
                        
                        # Replace the run_training_evo method with our custom version
                        experiment_manager.run_training_evo = custom_run_training_evo.__get__(experiment_manager)
                        
                        # Save the configuration
                        config_path = experiment_manager.config_dir / "config.pkl"
                        with open(config_path, "wb") as f:
                            pickle.dump(modified_config, f)
                        
                        json_path = experiment_manager.config_dir / "config.json"
                        with open(json_path, "w") as f:
                            # Convert numpy arrays to lists if present
                            config_json = {}
                            for key, value in modified_config.items():
                                if isinstance(value, np.ndarray):
                                    config_json[key] = value.tolist()
                                else:
                                    config_json[key] = value
                            json.dump(config_json, f, indent=4)
                        
                        print(f"Configuration saved to {config_path} and {json_path}")
                        
                        # Run the training
                        try:
                            print(f"Starting training for config={config_name}, hidden_size={hidden_size}, input_size={input_size}, seed={seed}")
                            results = experiment_manager.run_training_evo(verbose=verbose)
                            
                            experiment_result = {
                                "config_name": config_name,
                                "hidden_size": hidden_size,
                                "input_size": input_size,
                                "seed": seed,
                                "success": True,
                                "best_loss": float(results["best_loss"]),
                                "experiment_duration_minutes": float(results["experiment_duration_minutes"])
                            }
                            print(f"Training completed successfully with best loss: {results['best_loss']}")
                        except Exception as e:
                            import traceback
                            print(f"Error running experiment: {str(e)}")
                            print(traceback.format_exc())
                            experiment_result = {
                                "config_name": config_name,
                                "hidden_size": hidden_size,
                                "input_size": input_size,
                                "seed": seed,
                                "success": False,
                                "error": str(e)
                            }
                        
                        all_results["experiments"].append(experiment_result)
                        
                        # Clear GPU memory between runs
                        experiment_manager.clear_gpu_memory()
        
        # Restore original configuration
        self.config = original_config
        self.config_name = original_config_name
        self.timestamp = original_timestamp
        
        # Save overall results
        results_path = grid_search_dir / "grid_search_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"="*60)
        print(f"Grid search completed.")
        print(f"Results saved in: {grid_search_dir}")
        print(f"Summary saved to: {results_path}")
        print(f"="*60)
        
        return all_results
    
    def find_fixed_points(self, num_initial_states=100, seq_len=None, dynamics_mode="recurrence_only", model_path=None, plot=True, plot_traj=True, num_traj=10, plot_unique=True):
        """
        Find fixed points in the latent space of the trained RNN and optionally plot them.

        Args:
            num_initial_states (int): Number of initial states to use for optimization (default: 100).
            seq_len (int, optional): Length of sequences to generate for initial states.
                                    If None, uses config["seq_len"].
            dynamics_mode (str): Dynamics mode for sequence generation ('recurrence_only' for autonomous dynamics).
            model_path (str, optional): Path to a specific model.pth file to load.
                                        If None, loads the most recent model from the experiment's models directory.
            plot (bool): Whether to generate and save a plot of the fixed points (default: True).
            plot_traj (bool): Whether to include RNN state trajectories in the plot (default: True).
            num_traj (int): Number of trajectories to plot if plot_traj is True (default: 10).

        Returns:
            FixedPoints: Object containing the identified fixed points and metadata.

        Raises:
            FileNotFoundError: If no trained model is found or the specified model_path does not exist.
        """
        # Use config sequence length if not specified
        if seq_len is None:
            seq_len = self.config["seq_len"]

        # Determine the model file to load
        if model_path:
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Specified model path does not exist: {model_path}")
        else:
            # Load the most recent trained RNN model from the experiment's models directory
            model_files = list(self.models_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError("No trained model found in the models directory.")
            model_file = max(model_files, key=os.path.getmtime)

        # Initialize RNN with config parameters
        rnn = RNN(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            output_size=self.config["outputs"],
            biased=self.config["biased"]
        )
        rnn.load_model(str(model_file))

        # Generate sequences to extract initial states
        time_steps = num_initial_states * seq_len
        rnn_data = rnn.gen_seq(time_steps, dynamics_mode)
        hidden_states = rnn_data["h"]  # Shape: (time_steps, hidden_size)

        # Randomly select initial states
        indices = np.random.choice(time_steps, num_initial_states, replace=False)
        initial_states = hidden_states[indices]  # Shape: (num_initial_states, hidden_size)

        # Set inputs to zero for autonomous dynamics
        input_dim = self.config["input_size"]
        inputs = np.zeros((num_initial_states, input_dim))

        # Initialize FixedPointFinder with the internal nn.RNN module
        fpf = FixedPointFinderTorch(rnn.rnn)

        # Find fixed points
        rnn.train()
        all_fps, unique_fps = fpf.find_fixed_points(initial_states, inputs)

        # Save all fixed points to a file
        fps_path = self.config_dir / "fixed_points_all.pkl"
        with open(fps_path, "wb") as f:
            pickle.dump(all_fps, f)
        print(f"All fixed points saved to {fps_path}")

        # Save unique fixed points to a file
        unique_fps_path = self.config_dir / "fixed_points_unique.pkl"
        with open(unique_fps_path, "wb") as f:
            pickle.dump(unique_fps, f)
        print(f"Unique fixed points saved to {unique_fps_path}")

        # Plotting logic
        if plot:
            # Optionally generate state trajectories for plotting
            if plot_traj:
                # Generate trajectories: [num_traj, seq_len, hidden_size]
                traj_data = rnn.gen_seq(num_traj * seq_len, dynamics_mode)
                state_traj = traj_data["h"].reshape(num_traj, seq_len, self.config["hidden_size"])
            else:
                state_traj = None

            # Choose which fixed points to plot
            fps_to_plot = unique_fps if plot_unique else all_fps

            # Generate the plot using plot_fps
            fig = plot_fps(fps_to_plot, state_traj=state_traj, plot_batch_idx=list(range(num_traj)) if plot_traj else None)

            # Save the plot
            plot_name = "fixed_points_unique_plot.pdf" if plot_unique else "fixed_points_all_plot.pdf"
            plot_path = self.figs_path / plot_name
            fig.savefig(str(plot_path))
            print(f"Fixed points plot saved to {plot_path}")

        # Return both all_fps and unique_fps
        return all_fps, unique_fps

def rank_models(experiments):
    rankings = {}
    
    # Get unique configuration names
    config_names = set(exp['config_name'] for exp in experiments)
    
    for config_name in config_names:
        # Filter experiments for this config
        config_experiments = [exp for exp in experiments if exp['config_name'] == config_name]
        
        # Group successful experiments by (hidden_size, input_size)
        groups = defaultdict(list)
        for exp in config_experiments:
            if exp['success']:
                key = (exp['hidden_size'], exp['input_size'])
                groups[key].append(exp['best_loss'])
        
        # Calculate average and std of loss for each group
        averages = {}
        for key, losses in groups.items():
            if losses:  # Ensure there’s at least one successful experiment
                avg_loss = sum(losses) / len(losses)
                if len(losses) > 1:
                    # Calculate std: sqrt(sum((x - mean)^2) / N)
                    variance = sum((x - avg_loss) ** 2 for x in losses) / len(losses)
                    std_loss = math.sqrt(variance)
                else:
                    std_loss = 0.0  # Std is undefined for a single value
                averages[key] = {'average_loss': avg_loss, 'std_loss': std_loss}
        
        # Sort by average loss (ascending, since lower is better)
        sorted_keys = sorted(averages.keys(), key=lambda k: averages[k]['average_loss'])
        
        # Build ranking list with hidden_size, input_size, average_loss, and std_loss
        ranking_list = [
            {
                'hidden_size': k[0],
                'input_size': k[1],
                'average_loss': averages[k]['average_loss'],
                'std_loss': averages[k]['std_loss']
            }
            for k in sorted_keys
        ]
        
        rankings[config_name] = ranking_list
    
    return rankings


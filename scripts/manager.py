import os, torch, pickle, geomloss, json, datetime, random
import numpy as np
from pathlib import Path
from scripts.hmm import HMM
from scripts.rnn import RNN
from scripts.test import Test
from scripts.reverse import Reverse
import scripts.config

class Manager:
    """Class for managing experiment execution, data handling, and result storage."""
    def __init__(self, config_dict=None, config_name=None):
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
        #with open(config_path, "wb") as f:
            #pickle.dump(self.config, f)
            
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
        """Load a configuration from a file."""
        path = Path(path)
        if path.suffix == '.json':
            with open(path, "r") as f:
                self.config = json.load(f)
        else:
            with open(path, "rb") as f:
                self.config = pickle.load(f)
        print(f"Configuration loaded from {path}")
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent memory overload between experiments."""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    
    def run_training(self, verbose=False):
        """Train an RNN model on HMM data."""
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
            biased=self.config["biased"],
            weight_decay=self.config.get("weight_decay", 0.0)
        )
        
        # Load pre-trained weights if specified
        pretrained_path = self.config.get("pretrained_model_path")
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"Loading pre-trained model from {pretrained_path}")
                try:
                    rnn.load_model(pretrained_path)
                    print("Pre-trained model loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load pre-trained model: {e}")
                    print("Continuing with random initialization")
            else:
                print(f"Warning: Pre-trained model path '{pretrained_path}' does not exist")
                print("Continuing with random initialization")
        
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
        """Run tests on the trained model."""
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
        """Run reverse-engineering analysis on the trained model."""
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
        """Run the complete experiment pipeline."""
        # Record start time
        start_time = datetime.datetime.now()
        print(f"Starting experiment at {start_time}")
        
        # Save configuration
        self.save_config()
        
        # Train RNN model
        hmm, rnn, data_splits = self.run_training(verbose=verbose)
        
        # Run tests
        test_results = self.run_tests(hmm, rnn)
        
        # Run PCA analysis
        pca_results = self.run_reverse(rnn)
        
        # Record end time
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        
        # Save experiment results
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
            
        #print(f"Experiment results saved to {results_path}")
        #print(f"Experiment summary saved to {summary_path}")
        print(f"Experiment completed in {duration:.2f} minutes")
        
        return results
    
    def run_multiple_experiments(self, config_names, verbose=False):
        """Run multiple experiments with different configurations."""
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
        """Train an RNN model on HMM data, saving intermediate models every epochs. Tests and plots are generated only for the final model."""
        # Record start time
        start_time = datetime.datetime.now()
        print(f"Starting RNN training with evolution saving at {start_time}")

        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed(0)

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
        
        #print(f"Experiment results saved to {results_path}")
        #print(f"Experiment summary saved to {summary_path}")
        print(f"Experiment completed in {duration:.2f} minutes")
        
        # Clear GPU memory
        self.clear_gpu_memory()
        
        return results

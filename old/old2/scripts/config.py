"""
Configuration module for HMM-RNN experiments.
"""

class DefaultConfig:
    """Default configuration for experiments"""
    
    # HMM Parameters
    HMM = {
        "states": 3,
        "outputs": 3,
        "stay_prob": 0.95,
        "emission_method": "gaussian"  # Added emission_method parameter
    }
    
    # Data Generation Parameters
    DATA = {
        "num_seq": 30000,
        "seq_len": 150
    }
    
    # RNN Model Parameters
    RNN = {
        "input_size": 100,
        "hidden_size": 150,
        "num_layers": 1,
        "biased": [False, False]  # [rnn_bias, fc_bias]
    }
    
    # Training Parameters
    TRAINING = {
        "batch_size": 4096,
        "epochs": 1000,
        "learning_rates": [0.005, 0.001, 0.0001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": True  # Initialize hidden state with noise
    }
    
    # Paths
    PATHS = {
        "figs_path": "figs/",
        "data_path": "data/",
        "models_path": "models/"
    }
    
    @classmethod
    def get_config(cls):
        """Get the full configuration as a dictionary"""
        config = {
            # HMM config
            "states": cls.HMM["states"],
            "outputs": cls.HMM["outputs"],
            "stay_prob": cls.HMM["stay_prob"],
            "emission_method": cls.HMM["emission_method"],  # Added emission_method
            
            # Data config
            "num_seq": cls.DATA["num_seq"],
            "seq_len": cls.DATA["seq_len"],
            
            # RNN config
            "input_size": cls.RNN["input_size"],
            "hidden_size": cls.RNN["hidden_size"],
            "num_layers": cls.RNN["num_layers"],
            "biased": cls.RNN["biased"],
            
            # Training config
            "batch_size": cls.TRAINING["batch_size"],
            "epochs": cls.TRAINING["epochs"],
            "learning_rates": cls.TRAINING["learning_rates"],
            "tau": cls.TRAINING["tau"],
            "grad_clip": cls.TRAINING["grad_clip"],
            "init": cls.TRAINING["init"],
            
            # Paths
            "figs_path": cls.PATHS["figs_path"],
            "data_path": cls.PATHS["data_path"],
            "models_path": cls.PATHS["models_path"]
        }
        return config


class HMMSmall(DefaultConfig):
    """Configuration for a small HMM experiment (3 states, 3 outputs)"""
    HMM = {
        "states": 3,
        "outputs": 3,
        "stay_prob": 0.95,
        "emission_method": "gaussian"  # Added emission_method parameter
    }
    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,  # Reduced epochs for faster execution
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": True
    }


class HMMMedium(DefaultConfig):
    """Configuration for a medium HMM experiment (10 states, 3 outputs)"""
    HMM = {
        "states": 10,
        "outputs": 3,
        "stay_prob": 0.95,
        "emission_method": "gaussian"  # Added emission_method parameter
    }


class HMMLarge(DefaultConfig):
    """Configuration for a large HMM experiment (20 states, 3 outputs)"""
    HMM = {
        "states": 20,
        "outputs": 3,
        "stay_prob": 0.95,
        "emission_method": "gaussian"  # Added emission_method parameter
    }


class HMMComplexOutput(DefaultConfig):
    """Configuration for an HMM with more complex outputs (10 states, 5 outputs)"""
    HMM = {
        "states": 10,
        "outputs": 5,
        "stay_prob": 0.95,
        "emission_method": "gaussian"  # Added emission_method parameter
    }


class HMMStayProb80(DefaultConfig):
    """Configuration with low stay probability (80%)"""
    HMM = {
        "states": 10,
        "outputs": 3,
        "stay_prob": 0.80,
        "emission_method": "gaussian"
    }


class HMMStayProb99(DefaultConfig):
    """Configuration with high stay probability (99%)"""
    HMM = {
        "states": 10,
        "outputs": 3,
        "stay_prob": 0.99,
        "emission_method": "gaussian"
    }


class HMMLinearEmission(DefaultConfig):
    """Configuration with linear emission method"""
    HMM = {
        "states": 10,
        "outputs": 3,
        "stay_prob": 0.95,
        "emission_method": "linear"
    }


class HMMLowHidden(DefaultConfig):
    """Configuration with smaller hidden layer"""
    HMM = {
        "states": 10,
        "outputs": 3,
        "stay_prob": 0.95,
        "emission_method": "gaussian"
    }
    RNN = {
        "input_size": 100,
        "hidden_size": 50,  # Small hidden layer
        "num_layers": 1,
        "biased": [False, False]
    }

import numpy as np

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
        "target_prob": 0.05,
        "transition_method": "stay_prob",
        "emission_method": "gaussian",
        "custom_transition_matrix": None,
        "custom_emission_matrix": None 
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
    
    @classmethod
    def get_config(cls):
        """Get the full configuration as a dictionary"""
        config = {
            # HMM config
            "states": cls.HMM["states"],
            "outputs": cls.HMM["outputs"],
            "stay_prob": cls.HMM["stay_prob"],
            "target_prob": cls.HMM["target_prob"],
            "transition_method": cls.HMM["transition_method"],
            "emission_method": cls.HMM["emission_method"],
            "custom_transition_matrix": cls.HMM.get("custom_transition_matrix", None),
            "custom_emission_matrix": cls.HMM.get("custom_emission_matrix", None),
            
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
            "init": cls.TRAINING["init"]
        }
        return config

class HMMTwo(DefaultConfig):
    HMM = {
        "states": 2,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500, 
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": True
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 100
    }

class HMMThree(DefaultConfig):
    """Configuration for a small HMM experiment (3 states, 3 outputs)"""
    HMM = {
        "states": 3,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,  # Reduced epochs for faster execution
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": True
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMFour(DefaultConfig):
    """Configuration for a small HMM experiment (4 states, 3 outputs)"""
    HMM = {
        "states": 4,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,  # Reduced epochs for faster execution
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": True
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMFive(DefaultConfig):
    """Configuration for a small HMM experiment (5 states, 3 outputs)"""
    HMM = {
        "states": 5,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 1500,  # Reduced epochs for faster execution
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": True
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 40
    }
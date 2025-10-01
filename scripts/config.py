"""Configuration module for HMM-RNN experiments."""

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
        "init": True,
        "weight_decay": 0.0,  # L2 regularization strength
        "pretrained_model_path": None  # Path to pre-trained model weights
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
            "init": cls.TRAINING["init"],
            "weight_decay": cls.TRAINING["weight_decay"],
            "pretrained_model_path": cls.TRAINING.get("pretrained_model_path", None)
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
        "init": False
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 100
    }

class HMMThree(DefaultConfig):
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
        "epochs": 500, 
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMFour(DefaultConfig):
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
        "epochs": 500, 
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMFive(DefaultConfig):
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
        "epochs": 500,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 40
    }

class HMMThreeCycle(DefaultConfig):
    HMM = {
        "states": 3,
        "outputs": 3,
        "stay_prob": 0.99,
        "target_prob": 0.01,
        "transition_method": "fully",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 1500,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.4,
        "init": False,
        "weight_decay": 0
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 35
    }

class HMMThreeTriangular(DefaultConfig):
    HMM = {
        "states": 3,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "triangular"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMThreeTriangularFully(DefaultConfig):
    HMM = {
        "states": 3,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "fully",
        "emission_method": "triangular"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 1000,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.3,
        "init": False,
        "weight_decay": 0 
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMReview2(DefaultConfig):
    HMM = {
        "states": 2,
        "outputs": 2,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear",
        "custom_emission_matrix": [[1.0, 0.0], [0.0, 1.0]],
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False,
        "weight_decay": 0 
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 120
    }

class HMMReview(DefaultConfig):
    HMM = {
        "states": 4,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear",
        "custom_transition_matrix": [[0.94, 0.02, 0.02, 0.02], [0.02, 0.94, 0.02, 0.02], [0.02, 0.02, 0.94, 0.02], [0.02, 0.02, 0.02, 0.94]],
        "custom_emission_matrix": [[1, 0.0, 0.0, 0.0], [0.0, 1, 0.0, 0.0], [0.0, 0.0, 1, 0.0], [0.0, 0.0, 0.0, 1]],
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,
        "hidden_size": 150,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False,
        "weight_decay": 0 
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class HMMCycle4(DefaultConfig):
    HMM = {
        "states": 4,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear",
        "custom_transition_matrix": [[0.9, 0.05, 0.0, 0.05], [0.05, 0.9, 0.05, 0.0], [0.0, 0.05, 0.9, 0.05], [0.05, 0.0, 0.05, 0.9]],
        #'custom_transition_matrix': [[0.5, 0.25, 0.0, 0.0, 0.25], [0.25, 0.5, 0.25, 0.0, 0.0], [0.0, 0.25, 0.5, 0.25, 0.0], [0.0, 0.0, 0.25, 0.5, 0.25], [0.25, 0.0, 0.0, 0.25, 0.5]],
        "custom_emission_matrix": [[1, 0, 0], [0.6, 0, 0.4], [0, 0, 1], [0.4, 0, 0.6]],
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,
        "hidden_size": 150,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False,
        "weight_decay": 0 
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 30
    }

class pisellino4(DefaultConfig):
    HMM = {
        "states": 4,
        "outputs": 3,
        "stay_prob": 0.95,
        "target_prob": 0.05,
        "transition_method": "target_prob",
        "emission_method": "linear",
        "custom_transition_matrix": [[0.95, 0.025, 0.0, 0.025], [0.025, 0.95, 0.025, 0.0], [0.0, 0.025, 0.95, 0.025], [0.025, 0.0, 0.025, 0.95]],
        #'custom_transition_matrix': [[0.5, 0.25, 0.0, 0.0, 0.25], [0.25, 0.5, 0.25, 0.0, 0.0], [0.0, 0.25, 0.5, 0.25, 0.0], [0.0, 0.0, 0.25, 0.5, 0.25], [0.25, 0.0, 0.0, 0.25, 0.5]],
        "custom_emission_matrix": [[0.9, 0.1, 0], [0.1, 0.9, 0], [0, 0.1, 0.9], [0, 0.9, 0.1]],
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500,
        "hidden_size": 150,
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.3,
        "init": False,
        "weight_decay": 0 
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 40
    }

class HMMReview4(DefaultConfig):
    HMM = {
        "states": 2,
        "outputs": 3,
        "stay_prob": 0.85,
        "target_prob": 0.15,
        "transition_method": "target_prob",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 500, 
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False,
        "weight_decay": 0 
    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 50
    }

class HMMReview4_pretrained(DefaultConfig):
    HMM = {
        "states": 2,
        "outputs": 3,
        "stay_prob": 0.85,
        "target_prob": 0.15,
        "transition_method": "target_prob",
        "emission_method": "linear"
    }

    TRAINING = {
        "batch_size": 4096,
        "epochs": 100, 
        "learning_rates": [0.001],
        "tau": 1.0,
        "grad_clip": 0.9,
        "init": False,
        "weight_decay": 0,
        "pretrained_model_path": "TrainedModels/ReverseEngineeredModel/2HMM_3Outputs_linear_30kData_0.001lr_10.0Loss.pth"

    }

    DATA = {
        "num_seq": 30000,
        "seq_len": 50
    }
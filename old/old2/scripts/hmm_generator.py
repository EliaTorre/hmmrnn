import numpy as np
import torch
import torch.nn.functional as F
from hmmlearn.hmm import CategoricalHMM

class HMMGenerator:
    """
    Class for generating and managing Hidden Markov Model sequences.
    """
    def __init__(self, states, outputs, stay_prob=0.95, emission_method='linear'):
        """
        Initialize the HMM generator.
        
        Args:
            states (int): Number of hidden states
            outputs (int): Number of possible output symbols
            stay_prob (float): Probability of staying in the same state
            emission_method (str): Method for generating emission probabilities ('linear' or 'gaussian')
        """
        self.states = states
        self.outputs = outputs
        self.stay_prob = stay_prob
        self.emission_method = emission_method
        
        # Generate model parameters
        self.start_probabilities = self.generate_starting_probabilities()
        self.transition_matrix = self.generate_transition_matrix()
        self.emission_probabilities = self.generate_emission_probabilities(method=emission_method)
    
    def generate_starting_probabilities(self):
        """Generate uniform starting probabilities for HMM states"""
        return np.full(self.states, 1/self.states)
    
    def generate_transition_matrix(self):
        """Generate a tridiagonal transition matrix with specified stay probability"""
        transition_matrix = np.zeros((self.states, self.states))
        change_prob = 1 - self.stay_prob
        
        # Set probabilities for first and last states
        transition_matrix[0, 0] = self.stay_prob
        transition_matrix[0, 1] = change_prob
        transition_matrix[self.states-1, self.states-1] = self.stay_prob
        transition_matrix[self.states-1, self.states-2] = change_prob
        
        # Set probabilities for middle states
        for i in range(1, self.states-1):
            transition_matrix[i, i] = self.stay_prob
            transition_matrix[i, i-1] = change_prob/2
            transition_matrix[i, i+1] = change_prob/2
            
        # Normalize rows to ensure they sum to 1
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def generate_emission_probabilities(self, method='linear'):
        """
        Generate emission probabilities for an HMM using one of two methods.
        
        Parameters:
        - method: str, either 'linear' or 'gaussian'. 
                  'linear' interpolates between [0.99, 0.01, 0] and [0, 0.01, 0.99].
                  'gaussian' uses a Gaussian-like distribution over outputs.
                  
        Returns:
        - emission_probabilities: numpy array of shape (self.states, self.outputs)
        """
        if method == 'linear':
            emission_probabilities = np.zeros((self.states, self.outputs))
            for i in range(self.states):
                # Calculate interpolation factor (i ranges from 0 to num_states-1)
                alpha = i / (self.states - 1)
                p1 = 0.99 * (1 - alpha)
                p2 = 0.01
                p3 = 0.99 * alpha
                if self.outputs == 3:
                    emission_probabilities[i, :] = [p1, p2, p3]
                else:
                    # Handle case where outputs != 3
                    # Distribute probabilities across outputs using a linear interpolation
                    emission_probabilities[i, :] = np.linspace(p1, p3, self.outputs)
            return emission_probabilities
        elif method == 'gaussian':
            emission_probabilities = np.zeros((self.states, self.outputs))
            # Calculate center for each state, evenly spaced between 0 and outputs - 1
            centers = np.linspace(0, self.outputs - 1, num=self.states)
            for i in range(self.states):
                center = centers[i]
                for j in range(self.outputs):
                    distance = np.abs(j - center)
                    emission_probabilities[i, j] = np.exp(-0.2 * (distance / (0.1 * (self.outputs - 1)))**2)
            # Normalize each state's probabilities to sum to 1
            emission_probabilities /= emission_probabilities.sum(axis=1, keepdims=True)
            return emission_probabilities
        else:
            raise ValueError("Invalid method. Choose 'linear' or 'gaussian'.")
    
    def generate_sequences(self, num_seq, seq_len):
        """
        Generate sequences from the HMM.
        
        Args:
            num_seq (int): Number of sequences to generate
            seq_len (int): Length of each sequence
            
        Returns:
            tuple: (one_hot_sequences, sampled_states)
        """
        # Initialize HMM model
        model = CategoricalHMM(n_components=self.states)
        model.startprob_ = self.start_probabilities
        model.transmat_ = self.transition_matrix
        model.emissionprob_ = self.emission_probabilities
        
        # Generate sequences
        sampled_sequences = np.zeros((num_seq, seq_len))
        sampled_states = np.zeros((num_seq, seq_len))
        
        for i in range(num_seq):
            observations, hidden_states = model.sample(seq_len)
            sampled_sequences[i] = observations.reshape((seq_len))
            sampled_states[i] = hidden_states
            
        # One-hot encode the observation sequences
        one_hot_sequences = F.one_hot(torch.tensor(sampled_sequences).long(), num_classes=self.outputs)
        
        return one_hot_sequences, sampled_states
    
    def split_data(self, one_hot_sequences, sampled_states, train_ratio=1/3, val_ratio=1/3):
        """
        Split the generated data into training, validation, and test sets.
        
        Args:
            one_hot_sequences (torch.Tensor): One-hot encoded observation sequences
            sampled_states (np.ndarray): Hidden state sequences
            train_ratio (float): Fraction of data to use for training
            val_ratio (float): Fraction of data to use for validation
            
        Returns:
            dict: Dictionary containing train, validation, and test splits
        """
        num_seq = one_hot_sequences.shape[0]
        
        # Compute split indices
        train_end = int(num_seq * train_ratio)
        val_end = train_end + int(num_seq * val_ratio)
        
        # Split sequences
        train_seq = one_hot_sequences[:train_end]
        val_seq = one_hot_sequences[train_end:val_end]
        test_seq = one_hot_sequences[val_end:]
        
        # Split states
        train_states = sampled_states[:train_end]
        val_states = sampled_states[train_end:val_end]
        test_states = sampled_states[val_end:]
        
        return {
            "train_seq": train_seq,
            "val_seq": val_seq,
            "test_seq": test_seq,
            "train_states": train_states,
            "val_states": val_states,
            "test_states": test_states
        }
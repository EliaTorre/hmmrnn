import numpy as np
import torch
import torch.nn.functional as F
from hmmlearn.hmm import CategoricalHMM

class HMM:
    """
    Class for generating and managing Hidden Markov Model sequences.
    """
    def __init__(self, states, outputs, stay_prob=0.95, target_prob=0.05, 
             transition_method='target_prob', emission_method='linear',
             custom_transition_matrix=None, custom_emission_matrix=None):
        """
        Initialize the HMM generator.
        
        Args:
            states (int): Number of hidden states
            outputs (int): Number of possible output symbols
            stay_prob (float): Probability of staying in the same state
            target_prob (float): Target probability for transitions (when using target_prob method)
            transition_method (str): Method for generating transition matrix ('stay_prob', 'target_prob', or 'fully')
            emission_method (str): Method for generating emission probabilities ('linear' or 'gaussian')
            custom_transition_matrix (np.ndarray, optional): Custom transition matrix to use instead of generating one
            custom_emission_matrix (np.ndarray, optional): Custom emission matrix to use instead of generating one
        """
        self.states = states
        self.outputs = outputs
        self.stay_prob = stay_prob
        self.target_prob = target_prob
        self.transition_method = transition_method
        self.emission_method = emission_method
        self.custom_transition_matrix = custom_transition_matrix
        self.custom_emission_matrix = custom_emission_matrix
        
        # Generate model parameters
        self.start_probabilities = self.gen_start_prob()
        
        # Use custom matrices if provided, otherwise generate them
        if self.custom_transition_matrix is not None:
            print("Using custom transition matrix")
            self.transition_matrix = self.custom_transition_matrix
        else:
            self.transition_matrix = self.gen_trans_mat()
            
        if self.custom_emission_matrix is not None:
            print("Using custom emission matrix")
            self.emission_probabilities = self.custom_emission_matrix
        else:
            self.emission_probabilities = self.gen_emission_prob()
    
    def gen_start_prob(self):
        """Generate uniform starting probabilities for HMM states"""
        return np.full(self.states, 1/self.states)
    
    def gen_trans_mat(self):
        """
        Generate a transition matrix based on the chosen method.
        
        Methods:
            'stay_prob': Creates a tridiagonal matrix where each state is connected to its neighbors.
            'target_prob': Uses a target probability for uniform distribution.
            'fully': Creates a fully connected transition matrix where each state can transition to every other state.
        
        Returns:
            np.ndarray: Transition matrix with shape (states, states)
        """
        transition_matrix = np.zeros((self.states, self.states))
        
        if self.transition_method == 'stay_prob':
            # Method 1: Using stay probability
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
            
        elif self.transition_method == 'target_prob':
            # Method 2: Using target probability for uniform distribution
            q = self.target_prob ** (1/(self.states-1))
            
            # Set probabilities for first state
            transition_matrix[0, 0] = 1 - q
            transition_matrix[0, 1] = q
            
            # Set probabilities for last state
            transition_matrix[self.states-1, self.states-1] = 1 - q
            transition_matrix[self.states-1, self.states-2] = q
            
            # Set probabilities for middle states
            for i in range(1, self.states-1):
                transition_matrix[i, i] = 1 - 2*q
                transition_matrix[i, i-1] = q
                transition_matrix[i, i+1] = q
            
            # No normalization needed as each row already sums to 1
        
        elif self.transition_method == 'fully':
            # Method 3: Fully connected transition matrix
            # Every state can transition to every other state
            
            # Handle special case of only one state
            if self.states == 1:
                transition_matrix[0, 0] = 1.0
            else:
                change_prob = (1 - self.stay_prob) / (self.states - 1)
                
                # Set all transition probabilities
                for i in range(self.states):
                    for j in range(self.states):
                        if i == j:
                            # Probability of staying in the same state
                            transition_matrix[i, j] = self.stay_prob
                        else:
                            # Probability of transitioning to any other state
                            transition_matrix[i, j] = change_prob
            
            # No normalization needed as each row already sums to 1
        
        else:
            raise ValueError("Method must be either 'stay_prob', 'target_prob', or 'fully'")
        
        print("Transition Matrix:")
        print(transition_matrix)
        return transition_matrix
    
    def gen_emission_prob(self):
        """
        Generate emission probabilities for an HMM using one of two methods.
        
        Returns:
            np.ndarray: Emission probability matrix with shape (states, outputs)
        """
        if self.emission_method == 'linear':
            emission_probabilities = np.zeros((self.states, self.outputs))
            for i in range(self.states):
                # Calculate interpolation factor (i ranges from 0 to num_states-1)
                alpha = i / (self.states - 1) if self.states > 1 else 0
                p1 = 0.99 * (1 - alpha)
                p2 = 0.01
                p3 = 0.99 * alpha
                if self.outputs == 3:
                    emission_probabilities[i, :] = [p1, p2, p3]
                else:
                    # Handle case where outputs != 3
                    # Distribute probabilities across outputs using a linear interpolation
                    emission_probabilities[i, :] = np.linspace(p1, p3, self.outputs)
            print("Emission Probabilities:")
            print(emission_probabilities)
            return emission_probabilities
        elif self.emission_method == 'gaussian':
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
            print("Emission Probabilities:")
            print(emission_probabilities)
            return emission_probabilities
        else:
            raise ValueError("Invalid method. Choose 'linear' or 'gaussian'.")
    
    def gen_seq(self, num_seq, seq_len):
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
            dict: Dictionary containing the split data
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
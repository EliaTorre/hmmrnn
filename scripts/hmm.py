import numpy as np
import torch
import torch.nn.functional as F
from hmmlearn.hmm import CategoricalHMM

class HMM:
    """Class for generating and managing Hidden Markov Model sequences."""
    def __init__(self, states, outputs, stay_prob=0.95, target_prob=0.05, 
             transition_method='target_prob', emission_method='linear',
             custom_transition_matrix=None, custom_emission_matrix=None):

        self.states = states
        self.outputs = outputs
        self.stay_prob = stay_prob
        self.target_prob = target_prob
        self.transition_method = transition_method
        self.emission_method = emission_method
        self.custom_transition_matrix = custom_transition_matrix
        self.custom_emission_matrix = custom_emission_matrix
        self.start_probabilities = self.gen_start_prob()
        
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
        return np.full(self.states, 1/self.states)
    
    def gen_trans_mat(self):
        transition_matrix = np.zeros((self.states, self.states))
        
        if self.transition_method == 'stay_prob':
            # Method 1: Using stay probability
            change_prob = 1 - self.stay_prob
            transition_matrix[0, 0] = self.stay_prob
            transition_matrix[0, 1] = change_prob
            transition_matrix[self.states-1, self.states-1] = self.stay_prob
            transition_matrix[self.states-1, self.states-2] = change_prob
            for i in range(1, self.states-1):
                transition_matrix[i, i] = self.stay_prob
                transition_matrix[i, i-1] = change_prob/2
                transition_matrix[i, i+1] = change_prob/2
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            
        elif self.transition_method == 'target_prob':
            # Method 2: Using target probability for uniform distribution
            q = self.target_prob ** (1/(self.states-1))
            transition_matrix[0, 0] = 1 - q
            transition_matrix[0, 1] = q
            transition_matrix[self.states-1, self.states-1] = 1 - q
            transition_matrix[self.states-1, self.states-2] = q
            for i in range(1, self.states-1):
                transition_matrix[i, i] = 1 - 2*q
                transition_matrix[i, i-1] = q
                transition_matrix[i, i+1] = q 
        
        elif self.transition_method == 'fully':
            # Method 3: Fully connected transition matrix
            if self.states == 1:
                transition_matrix[0, 0] = 1.0
            else:
                change_prob = (1 - self.stay_prob) / (self.states - 1)
                for i in range(self.states):
                    for j in range(self.states):
                        if i == j:
                            transition_matrix[i, j] = self.stay_prob
                        else:
                            transition_matrix[i, j] = change_prob
        else:
            raise ValueError("Method must be either 'stay_prob', 'target_prob', or 'fully'")
        print("Transition Matrix:")
        print(transition_matrix)
        return transition_matrix
    
    def gen_emission_prob(self):
        if self.emission_method == 'linear':
            emission_probabilities = np.zeros((self.states, self.outputs))
            for i in range(self.states):
                alpha = i / (self.states - 1) if self.states > 1 else 0
                p1 = 0.99 * (1 - alpha)
                p2 = 0.01
                p3 = 0.99 * alpha
                if self.outputs == 3:
                    emission_probabilities[i, :] = [p1, p2, p3]
                else:
                    emission_probabilities[i, :] = np.linspace(p1, p3, self.outputs)
            print("Emission Probabilities:")
            print(emission_probabilities)
            return emission_probabilities
        elif self.emission_method == 'gaussian':
            emission_probabilities = np.zeros((self.states, self.outputs))
            centers = np.linspace(0, self.outputs - 1, num=self.states)
            for i in range(self.states):
                center = centers[i]
                for j in range(self.outputs):
                    distance = np.abs(j - center)
                    emission_probabilities[i, j] = np.exp(-0.2 * (distance / (0.1 * (self.outputs - 1)))**2)
            emission_probabilities /= emission_probabilities.sum(axis=1, keepdims=True)
            print("Emission Probabilities:")
            print(emission_probabilities)
            return emission_probabilities
        else:
            raise ValueError("Invalid method. Choose 'linear' or 'gaussian'.")
    
    def gen_seq(self, num_seq, seq_len):
        """Generate sequences from the HMM."""
        model = CategoricalHMM(n_components=self.states)
        model.startprob_ = self.start_probabilities
        model.transmat_ = self.transition_matrix
        model.emissionprob_ = self.emission_probabilities
        
        sampled_sequences = np.zeros((num_seq, seq_len))
        sampled_states = np.zeros((num_seq, seq_len))
        
        for i in range(num_seq):
            observations, hidden_states = model.sample(seq_len)
            sampled_sequences[i] = observations.reshape((seq_len))
            sampled_states[i] = hidden_states
            
        one_hot_sequences = F.one_hot(torch.tensor(sampled_sequences).long(), num_classes=self.outputs)
        
        return one_hot_sequences, sampled_states
    
    def split_data(self, one_hot_sequences, sampled_states, train_ratio=1/3, val_ratio=1/3):
        """Split the generated data into training, validation, and test sets."""
        num_seq = one_hot_sequences.shape[0]
        train_end = int(num_seq * train_ratio)
        val_end = train_end + int(num_seq * val_ratio)
        train_seq = one_hot_sequences[:train_end]
        val_seq = one_hot_sequences[train_end:val_end]
        test_seq = one_hot_sequences[val_end:]
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
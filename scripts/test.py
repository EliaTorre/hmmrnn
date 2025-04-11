import numpy as np
import torch
import matplotlib.pyplot as plt
from scripts.sinkhorn import SinkhornSolver

class Test:
    """
    Class for testing and comparing HMM and RNN models.
    """
    def __init__(self, hmm, rnn, num_seq, seq_len, outputs):
        """
        Initialize the model tester.
        
        Args:
            hmm: HMM instance
            rnn: RNN instance
            num_seq (int): Number of sequences to test
            seq_len (int): Length of each sequence
            outputs (int): Number of output symbols
        """
        self.hmm = hmm
        self.rnn = rnn
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.outputs = outputs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Colors for visualization
        self.colors_dict = {0: 'red', 1: 'blue', 2: 'green'}
        if outputs > 3:
            # Generate more colors if needed
            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.values())
            self.colors_dict = {i: colors[i % len(colors)] for i in range(outputs)}
    
    def gen_test_data(self):
        """
        Generate test data from both HMM and RNN.
        
        Returns:
            dict: Dictionary containing HMM and RNN data
        """
        print("Generating HMM test sequences...")
        # Generate HMM sequences
        self.hmm_data, _ = self.hmm.gen_seq(self.num_seq, self.seq_len)
        self.hmm_data2, _ = self.hmm.gen_seq(self.num_seq, self.seq_len)
        self.hmm_data3, _ = self.hmm.gen_seq(self.num_seq, self.seq_len)
        
        print("Generating RNN test sequences...")
        # Generate RNN sequences using batch mode to get proper shape directly
        rnn_outputs = self.rnn.gen_seq(dynamics_mode="full", batch_mode=True, 
                                    num_seq=self.num_seq, seq_len=self.seq_len)
        self.rnn_data = rnn_outputs["outs"]
        
        return {
            "hmm_data": self.hmm_data,
            "rnn_data": self.rnn_data
        }
    
    def match(self, seq1, seq2):
        """
        Match similar sequences using Sinkhorn transport.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            tuple: (matched_seq1, matched_seq2)
        """
        sinkhorn = SinkhornSolver(epsilon=0.1, iterations=1000)
        
        # Flatten sequences
        seq1_flat = seq1.reshape(seq1.shape[0], -1)
        seq1_flat = torch.tensor(seq1_flat).float().to(self.device)
        
        seq2_flat = seq2.reshape(seq2.shape[0], -1)
        seq2_flat = torch.tensor(seq2_flat).float().to(self.device)
        
        # Calculate transport plan
        tp = sinkhorn(seq1_flat, seq2_flat)
        
        # Match sequences based on transport plan
        matched_seq1 = seq1_flat[tp[1].argmax(0)].cpu().detach().numpy()
        matched_seq2 = seq2_flat.cpu().detach().numpy()
        
        return matched_seq1, matched_seq2
    
    def euclidean_distances(self):
        """
        Calculate Euclidean distances between matched sequences.
        
        Returns:
            dict: Dictionary containing HMM and RNN distances
        """
        print("Calculating Euclidean distances...")
        # Match HMM with HMM for baseline
        hmm_matched1, hmm_matched2 = self.match(self.hmm_data2, self.hmm_data3)
        
        # Match HMM with RNN
        hmm_matched_rnn, rnn_matched_hmm = self.match(self.hmm_data, self.rnn_data)
        
        # Calculate distances
        hmm_distances = np.linalg.norm(hmm_matched1 - hmm_matched2, axis=1)
        rnn_distances = np.linalg.norm(hmm_matched_rnn - rnn_matched_hmm, axis=1)
        
        return {
            "hmm_distances": hmm_distances,
            "rnn_distances": rnn_distances
        }
    
    def volatilities(self, seq):
        """
        Calculate observation volatilities (frequency of state changes).
        
        Args:
            seq: Sequence data
            
        Returns:
            tuple: (mean_volatility, std_volatility)
        """
        seq_max = np.argmax(seq, axis=2)
        diff = np.diff(seq_max, axis=1)
        changes = np.count_nonzero(diff, axis=1)
        return np.mean(changes), np.std(changes)
    
    def frequencies(self, seq):
        """
        Calculate the frequency of each output in the sequences.
        
        Args:
            seq: Sequence data
            
        Returns:
            tuple: (mean_frequencies, std_frequencies)
        """
        frequencies = np.zeros((seq.shape[0], seq.shape[2]))
        seq_max = np.argmax(seq, axis=2)
        
        for i in range(seq.shape[0]):
            unique, counts = np.unique(seq_max[i], return_counts=True)
            frequencies[i, unique] = counts / sum(counts)
            
        return np.mean(frequencies, axis=0), np.std(frequencies, axis=0)
    
    def transition_matrices(self, seq):
        """
        Calculate transition matrices between states.
        
        Args:
            seq: Sequence data
            
        Returns:
            np.ndarray: Transition matrix
        """
        mat = np.zeros((self.outputs, self.outputs))
        seq_max = np.argmax(seq, axis=2)
        for i in range(seq.shape[0]):
            for j in range(1, self.seq_len):
                mat[seq_max[i, j-1], seq_max[i, j]] += 1
        return mat/mat.sum(axis=1, keepdims=True)

    def run_all(self):
        """
        Run all tests and return the results.
        
        Returns:
            dict: Dictionary containing all test results
        """
        print("Generating test data...")
        self.gen_test_data()
        
        print("Calculating Euclidean distances...")
        distances = self.euclidean_distances()
        
        print("Calculating volatilities...")
        hmm_volatilities = self.volatilities(self.hmm_data)
        rnn_volatilities = self.volatilities(self.rnn_data)
        
        print("Calculating frequencies...")
        hmm_freqs = self.frequencies(self.hmm_data)
        rnn_freqs = self.frequencies(self.rnn_data)
        
        print("Calculating transition matrices...")
        hmm_trans = self.transition_matrices(self.hmm_data)
        rnn_trans = self.transition_matrices(self.rnn_data)
        
        return {
            "distances": distances,
            "volatilities": {
                "hmm": hmm_volatilities,
                "rnn": rnn_volatilities
            },
            "frequencies": {
                "hmm": hmm_freqs,
                "rnn": rnn_freqs
            },
            "transitions": {
                "hmm": hmm_trans,
                "rnn": rnn_trans
            }
        }
    
    def gen_plots(self, results, save_path="", model_info=None):
        """
        Generate and save comparison plots.
        
        Args:
            results (dict): Results from run_all()
            save_path (str): Path to save plots
            model_info (dict, optional): Dictionary containing model information
        """
        print("Generating comparison plots...")
        
        # Extract model info if provided
        states_info = f"States: {model_info['states']}, " if model_info and 'states' in model_info else ""
        hidden_info = f"Hidden: {model_info['hidden_size']}, " if model_info and 'hidden_size' in model_info else ""
        input_info = f"Input: {model_info['input_size']}" if model_info and 'input_size' in model_info else ""
        model_info_str = f"{states_info}{hidden_info}{input_info}"
        
        # 1. Euclidean distance plot
        plt.figure(figsize=(5, 5))
        x_positions = [0, 0.5]
        hmm_distances = results["distances"]["hmm_distances"]
        rnn_distances = results["distances"]["rnn_distances"]
        
        # Use suptitle for model info to avoid overlap
        if model_info_str:
            plt.suptitle(model_info_str, fontsize=12, y=0.98)
        
        plt.errorbar(x_positions[0], np.mean(hmm_distances), 
                     yerr=np.std(hmm_distances), fmt='o', color='darkgreen', capsize=3)
        plt.errorbar(x_positions[1], np.mean(rnn_distances), 
                     yerr=np.std(rnn_distances), fmt='o', color='darkred', capsize=3)
        
        plt.xticks(x_positions, ["HMM", "RNN"], fontsize=10)
        for i, (mean, std) in enumerate(zip([np.mean(hmm_distances), np.mean(rnn_distances)],
                                           [np.std(hmm_distances), np.std(rnn_distances)])):
            plt.text(x_positions[i], mean + std + 0.1, f"{mean:.2f}", ha='center', fontsize=8)
            
        plt.title(f"Mean Euclidean Distances", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("Distance")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_path}/euclidean_distances.pdf")
        plt.close()
        
        # 2. Observation volatilities
        plt.figure(figsize=(5, 5))
        x_positions = [0, 0.5]
        hmm_vol = results["volatilities"]["hmm"]
        rnn_vol = results["volatilities"]["rnn"]
        
        # Use suptitle for model info to avoid overlap
        if model_info_str:
            plt.suptitle(model_info_str, fontsize=12, y=0.98)
        
        plt.errorbar(x_positions[0], hmm_vol[0], yerr=hmm_vol[1], 
                     fmt='o', color='darkred', capsize=3)
        plt.errorbar(x_positions[1], rnn_vol[0], yerr=rnn_vol[1], 
                     fmt='o', color='darkgreen', capsize=3)
        
        plt.xticks(x_positions, ["HMM", "RNN"], fontsize=10)
        for i, (mean, std) in enumerate(zip([hmm_vol[0], rnn_vol[0]], [hmm_vol[1], rnn_vol[1]])):
            plt.text(x_positions[i], mean + std + 1, f"{mean:.2f}", ha='center', fontsize=8)
            
        plt.title(f"Observation Volatilities", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("# of Changes in Outputs")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_path}/observation_volatilities.pdf")
        plt.close()
        
        # 3. Output frequencies
        bar_width = 0.4
        indices = np.arange(self.outputs)
        plt.figure(figsize=(9, 7))
        
        hmm_freqs = results["frequencies"]["hmm"]
        rnn_freqs = results["frequencies"]["rnn"]
        
        # Use suptitle for model info to avoid overlap
        if model_info_str:
            plt.suptitle(model_info_str, fontsize=12, y=0.98)
        
        plt.errorbar(indices - bar_width / 2, hmm_freqs[0], yerr=hmm_freqs[1], 
                     fmt='o', label="HMM", color="darkgreen", capsize=3)
        plt.errorbar(indices + bar_width / 2, rnn_freqs[0], yerr=rnn_freqs[1], 
                     fmt='o', label="RNN", color="darkred", capsize=3)
        
        for i, (hmm_value, hmm_err, rnn_value, rnn_err) in enumerate(
                zip(hmm_freqs[0], hmm_freqs[1], rnn_freqs[0], rnn_freqs[1])):
            plt.text(i - bar_width / 2, hmm_value + hmm_err + 0.01, 
                     f"{hmm_value:.2f}", fontsize=10, ha="center", va="bottom")
            plt.text(i + bar_width / 2, rnn_value + rnn_err + 0.01, 
                     f"{rnn_value:.2f}", fontsize=10, ha="center", va="bottom")
            
        plt.xticks(indices, [f"Output {i + 1}" for i in range(self.outputs)])
        plt.title(f"Output Frequencies", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{save_path}/output_frequencies.pdf")
        plt.close()
        
        # 4. Transition matrices
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Increased width to accommodate colorbar
        
        hmm_trans = results["transitions"]["hmm"]
        rnn_trans = results["transitions"]["rnn"]
        
        # Use suptitle for main title and model info with proper spacing
        plt.suptitle(f'Transition Matrices between Outputs', fontsize=16, fontweight='bold', y=0.98)
        if model_info_str:
            plt.figtext(0.5, 0.92, f"{model_info_str}", ha='center', fontsize=12)
        
        # Create a common colorbar for both matrices
        vmin = min(hmm_trans.min(), rnn_trans.min())
        vmax = max(hmm_trans.max(), rnn_trans.max())
        
        for ax, matrix, title in zip(axs, [hmm_trans, rnn_trans], ["HMM", "RNN"]):
            cax = ax.matshow(matrix, cmap="Blues", alpha=0.6, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
            ax.set_xticks(np.arange(self.outputs))
            ax.set_yticks(np.arange(self.outputs))
            ax.set_xticklabels(np.arange(1, self.outputs + 1))
            ax.set_yticklabels(np.arange(1, self.outputs + 1))
            
            for i in range(self.outputs):
                for j in range(self.outputs):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", va="center", ha="center")
        
        # Add colorbar with proper spacing
        #cbar = fig.colorbar(cax, ax=axs, shrink=0.8, pad=0.02)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 0.95, 0.9])
        plt.savefig(f"{save_path}/transition_matrices.pdf")
        plt.close()
        
        print(f"Plots saved to {save_path}")

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA

class Reverse:
    """
    Class for analyzing and visualizing RNN hidden states through PCA.
    """
    def __init__(self, rnn, num_seq, seq_len, outputs):
        """
        Initialize the PCA analyzer.
        
        Args:
            rnn: RNN instance
            num_seq (int): Number of sequences
            seq_len (int): Length of each sequence
            outputs (int): Number of output symbols
        """
        self.rnn = rnn
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.outputs = outputs
        self.time_steps = num_seq * seq_len
        
        # Colors for visualization
        self.colors_dict = {0: 'red', 1: 'blue', 2: 'green'}
        if outputs > 3:
            # Generate more colors if needed
            import matplotlib.colors as mcolors
            colors = list(mcolors.TABLEAU_COLORS.values())
            self.colors_dict = {i: colors[i % len(colors)] for i in range(outputs)}
            
    def run_pca(self, hidden_states, n_components=5):
        """
        Run PCA on hidden states.
        
        Args:
            hidden_states: Hidden state data
            n_components (int): Number of principal components to extract
            
        Returns:
            tuple: (pca, pca_result)
        """
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(hidden_states)
        
        return pca, pca_result
    
    def gen_data(self, dynamics_mode="full"):
        """
        Generate RNN data and perform PCA on hidden states.
        
        Args:
            dynamics_mode (str): Dynamics mode for RNN sequence generation
            
        Returns:
            dict: Dictionary with analysis results
        """
        # Generate RNN sequences
        print("Generating RNN sequences for PCA...")
        rnn_data = self.rnn.gen_seq(self.time_steps, dynamics_mode)
        
        # Extract outputs and hidden states
        self.rnn_outputs = np.argmax(rnn_data["outs"], axis=-1)
        self.rnn_hiddens = rnn_data["h"]
        
        # Run PCA
        print("Running PCA...")
        self.pca, self.pca_hiddens = self.run_pca(self.rnn_hiddens)
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return {
            "outputs": self.rnn_outputs,
            "hiddens": self.rnn_hiddens,
            "pca_hiddens": self.pca_hiddens,
            "explained_variance": self.pca.explained_variance_ratio_
        }
    
    def plot_2d(self, num_points=1000, save_path=None):
        """
        Create a 2D plot of the PCA-projected hidden states.
        
        Args:
            num_points (int): Number of points to plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            tuple: (fig, ax)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw arrows for transitions
        for i in range(min(num_points, len(self.pca_hiddens)-1)):
            color = self.colors_dict[self.rnn_outputs[i]]
            start = self.pca_hiddens[i, :2]
            end = self.pca_hiddens[i + 1, :2]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            ax.arrow(start[0], start[1], dx, dy, color=color,
                    head_width=0.4, head_length=0.4, length_includes_head=True)
        
        ax.set_title("PCA Projection of RNN Hidden States (2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"2D trajectory plot saved to {save_path}")
        
        return fig, ax
    
    def plot_3d(self, num_points=1000, save_path=None):
        """
        Create an interactive 3D plot of the PCA-projected hidden states.
        
        Args:
            num_points (int): Number of points to plot
            save_path (str, optional): Path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        fig = go.Figure()
        
        # Draw lines for transitions
        for i in range(min(num_points, len(self.pca_hiddens)-1)):
            color = self.colors_dict[self.rnn_outputs[i]]
            x0, y0, z0 = self.pca_hiddens[i, 0], self.pca_hiddens[i, 1], self.pca_hiddens[i, 2]
            x1, y1, z1 = self.pca_hiddens[i + 1, 0], self.pca_hiddens[i + 1, 1], self.pca_hiddens[i + 1, 2]
            
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=4, color=color),
                showlegend=False
            ))
        
        fig.update_layout(
            title="3D PCA Projection of RNN Hidden States",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"3D trajectory plot saved to {save_path}")
        
        return fig
    
    def run_analysis(self, dynamics_mode="full", save_path=""):
        """
        Run the full PCA analysis and generate plots.
        
        Args:
            dynamics_mode (str): Dynamics mode for RNN sequence generation
            save_path (str): Base path to save plots
            
        Returns:
            dict: Dictionary with analysis results
        """
        # Generate data and run PCA
        data = self.gen_data(dynamics_mode)
        
        # Create 2D plot
        print("Generating 2D PCA trajectory plot...")
        self.plot_2d(save_path=f"{save_path}/latent_trajectory_2d.pdf")
        
        # Create 3D plot
        print("Generating 3D PCA trajectory plot...")
        self.plot_3d(save_path=f"{save_path}/latent_trajectory_3d.html")
        
        return {
            "explained_variance": self.pca.explained_variance_ratio_,
            "data": data
        }
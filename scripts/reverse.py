import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA

class Reverse:
    def __init__(self, rnn, num_seq, seq_len, outputs):
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
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(hidden_states)
        
        return pca, pca_result
    
    def gen_data(self, dynamics_mode="full"):
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
    
    def plot_2d(self, num_points=1000, save_path=None, model_info=None):
        # Increase figure size to accommodate title and subtitle
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw arrows for transitions
        for i in range(min(num_points, len(self.pca_hiddens)-1)):
            color = self.colors_dict[self.rnn_outputs[i]]
            start = self.pca_hiddens[i, :2]
            end = self.pca_hiddens[i + 1, :2]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            ax.arrow(start[0], start[1], dx, dy, color=color,
                    head_width=0.4, head_length=0.4, length_includes_head=True)
        
        # Add model info as suptitle if provided
        if model_info:
            states_info = f"States: {model_info['states']}, " if 'states' in model_info else ""
            hidden_info = f"Hidden: {model_info['hidden_size']}, " if 'hidden_size' in model_info else ""
            input_info = f"Input: {model_info['input_size']}" if 'input_size' in model_info else ""
            model_info_str = f"{states_info}{hidden_info}{input_info}"
            plt.suptitle(model_info_str, fontsize=12, y=0.98)
            
        # Add title with padding
        ax.set_title("PCA Projection of RNN Hidden States (2D)", fontsize=16, fontweight="bold", pad=20)
            
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.grid(True)
        
        # Add padding to ensure no overlap
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path)
            print(f"2D trajectory plot saved to {save_path}")
        
        return fig, ax
    
    def plot_3d(self, num_points=1000, save_path=None, cone_scale=0.1, model_info=None):
        fig = go.Figure()
        
        # Draw lines for transitions
        for i in range(min(num_points, len(self.pca_hiddens)-1)):
            color = self.colors_dict[self.rnn_outputs[i]]
            
            # Current and next point coordinates
            x0, y0, z0 = self.pca_hiddens[i, 0], self.pca_hiddens[i, 1], self.pca_hiddens[i, 2]
            x1, y1, z1 = self.pca_hiddens[i + 1, 0], self.pca_hiddens[i + 1, 1], self.pca_hiddens[i + 1, 2]
            
            # Direction vector for cone
            u = x1 - x0
            v = y1 - y0
            w = z1 - z0
            
            # Add lines connecting points
            fig.add_trace(go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False
            ))
            
            # Add cone at the starting point pointing to the next point
            fig.add_trace(go.Cone(
                x=[x0],
                y=[y0],
                z=[z0],
                u=[u * cone_scale],
                v=[v * cone_scale],
                w=[w * cone_scale],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                sizemode="absolute",
                anchor="tail"
            ))
        
        # Create title with model info if provided
        title = "3D PCA Projection of RNN Hidden States"
        if model_info:
            states_info = f"States: {model_info['states']}, " if 'states' in model_info else ""
            hidden_info = f"Hidden: {model_info['hidden_size']}, " if 'hidden_size' in model_info else ""
            input_info = f"Input: {model_info['input_size']}" if 'input_size' in model_info else ""
            model_info_str = f"{states_info}{hidden_info}{input_info}"
            subtitle = f"<br><span style='font-size:12px; font-weight:normal'>{model_info_str}</span>"
            title = title + subtitle
        
        fig.update_layout(
            title=title,
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
    
    def run_analysis(self, dynamics_mode="full", save_path="", model_info=None):
        # Generate data and run PCA
        data = self.gen_data(dynamics_mode)
        
        # Create 2D plot
        print("Generating 2D PCA trajectory plot...")
        self.plot_2d(save_path=f"{save_path}/latent_trajectory_2d.pdf", model_info=model_info)
        
        # Create 3D plot
        print("Generating 3D PCA trajectory plot...")
        self.plot_3d(save_path=f"{save_path}/latent_trajectory_3d.html", model_info=model_info)
        
        return {
            "explained_variance": self.pca.explained_variance_ratio_,
            "data": data
        }

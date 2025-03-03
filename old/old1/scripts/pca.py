import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scripts.run import run
from scripts.config import *

def pca_projection(hidden_states, n_components=2):
    """
    Performs PCA on the hidden states keeping 5 components,
    then returns only the first n_components for plotting.
    """
    pca = PCA(n_components=5)
    pca_hidden_states = pca.fit_transform(hidden_states)
    return pca, pca_hidden_states[:, :n_components]

def matplotlib_plot(pca_hidden_states, rnn_outputs, colors_dict):
    """
    Plots a 2D trajectory of the PCA-projected hidden states using matplotlib.
    Arrows (drawn with ax.arrow) indicate transitions between timesteps.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # For example, draw arrows for the first 5000 segments.
    for i in range(1000):
        color = colors_dict[rnn_outputs[i]]
        start = pca_hidden_states[i]
        end = pca_hidden_states[i + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ax.arrow(start[0], start[1], dx, dy, color=color,
                 head_width=0.4, head_length=0.4, length_includes_head=True)
    
    ax.set_title("PCA Projection of RNN Hidden States (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    
    return fig, ax

def plotly_plot(pca_hidden_states, rnn_outputs, colors_dict):
    """
    Creates an interactive 3D plot of the PCA-projected hidden states using Plotly.
    Each consecutive pair of points is connected with a line segment (with markers) colored
    according to the provided colors_dict.
    """
    fig = go.Figure()
    
    # Iterate over consecutive points
    for i in range(1000):
        color = colors_dict[rnn_outputs[i]]
        x0, y0, z0 = pca_hidden_states[i, 0], pca_hidden_states[i, 1], pca_hidden_states[i, 2]
        x1, y1, z1 = pca_hidden_states[i + 1, 0], pca_hidden_states[i + 1, 1], pca_hidden_states[i + 1, 2]
        
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
    
    return fig

# Define color mapping
colors_dict = {0: 'red', 1: 'blue', 2: 'green'}

def projection(outputs, model, path):
    print("Generating RNN Data...")
    # Generate RNN sequences
    rnn_data = run(
        model_path=model,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_outputs=outputs,
        time_steps=time_steps,
        dynamics_mode="full"
    )
    # Determine the most likely output at each timestep
    rnn_outputs = np.argmax(rnn_data["outs"], axis=-1)
    rnn_hiddens = rnn_data["h"]

    # Create 2D plot with matplotlib using the first two PCs
    pca, pca_hidden_states_2d = pca_projection(rnn_hiddens, n_components=2)
    print("Explained variance ratio (top 5 PCs):", pca.explained_variance_ratio_)
    fig_matplotlib, ax_matplotlib = matplotlib_plot(pca_hidden_states_2d, rnn_outputs, colors_dict)
    pdf_file = path + "figs/latent_trajectory.pdf"
    fig_matplotlib.savefig(pdf_file)
    print(f"Matplotlib visualization saved as {pdf_file}")

    # Create interactive 3D plot with Plotly using the first three PCs
    _, pca_hidden_states_3d = pca_projection(rnn_hiddens, n_components=3)
    fig_plotly = plotly_plot(pca_hidden_states_3d, rnn_outputs, colors_dict)
    html_file = path + "figs/latent_trajectory.html"
    fig_plotly.write_html(html_file)
    print(f"Plotly visualization saved as {html_file}")

def main():
    print("Generating RNN Data...")
    # Generate RNN sequences
    rnn_data = run(
        model_path=model_path,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_outputs=outputs,
        time_steps=time_steps,
        dynamics_mode="full"
    )
    # Determine the most likely output at each timestep
    rnn_outputs = np.argmax(rnn_data["outs"], axis=-1)
    rnn_hiddens = rnn_data["h"]

    # Create 2D plot with matplotlib using the first two PCs
    pca, pca_hidden_states_2d = pca_projection(rnn_hiddens, n_components=2)
    print("Explained variance ratio (top 5 PCs):", pca.explained_variance_ratio_)
    fig_matplotlib, ax_matplotlib = matplotlib_plot(pca_hidden_states_2d, rnn_outputs, colors_dict)
    pdf_file = "figs/latent_trajectory.pdf"
    fig_matplotlib.savefig(pdf_file)
    print(f"Matplotlib visualization saved as {pdf_file}")

    # Create interactive 3D plot with Plotly using the first three PCs
    _, pca_hidden_states_3d = pca_projection(rnn_hiddens, n_components=3)
    fig_plotly = plotly_plot(pca_hidden_states_3d, rnn_outputs, colors_dict)
    html_file = "figs/latent_trajectory.html"
    fig_plotly.write_html(html_file)
    print(f"Plotly visualization saved as {html_file}")

if __name__ == "__main__":
    main()

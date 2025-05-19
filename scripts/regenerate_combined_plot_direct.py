import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import os
import seaborn as sns

def create_combined_grid_plot():
    """
    Create a 3x4 grid plot where rows are different plot types and columns are different models.
    This is a simplified version that directly regenerates the plot without re-running the pipeline.
    """
    # Create figure with 3x4 grid (rows are plot types, columns are models)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Define colors for each output
    colors = {0: 'darkgreen', 1: 'royalblue', 2: 'darkred'}
    
    # Define variance levels for contour plots
    variances = [0.1, 1.0, 2.0, 3.0, 4.0]
    
    # Define colors from the inferno colormap for contour plots
    contour_colors = plt.cm.inferno(np.linspace(0.9, 0.1, 5))
    
    # Column labels (model types)
    col_labels = ["HMMTwo", "HMMThree", "HMMFour", "HMMFive"]
    # Row labels (plot types)
    row_labels = ["Trajectories without input", "Trajectories with input", "Variance contours"]
    
    # For each subplot, load the corresponding image and display it
    for i in range(3):
        for j in range(4):
            model_type = col_labels[j]
            
            # Determine which image to load based on row and column
            if i == 0:  # First row: traj_no_in plots
                img_path = f"traj_no_in/{model_type}_h150_i100_no_in.png"
            elif i == 1:  # Second row: traj_in plots
                img_path = f"traj_in/{model_type}_h150_i100_in.png"
            else:  # Third row: contour plots
                img_path = f"variance_contours/{model_type}_h150_i100_var.png"
            
            # Check if the image exists
            if os.path.exists(img_path):
                # Load and display the image
                img = plt.imread(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')  # Turn off axis
            else:
                # If image doesn't exist, display a placeholder
                axes[i, j].text(0.5, 0.5, f"Image not found:\n{img_path}", 
                               ha='center', va='center', fontsize=10)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
    
    # Set column labels (model types)
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=16, fontweight='bold')
    
    # Set row labels
    for i, label in enumerate(row_labels):
        # Add row labels to the left of the first column
        fig.text(0.01, 0.75 - i*0.25, label, fontsize=16, fontweight='bold', 
                 ha='left', va='center', rotation=90)
    
    # Create legend for trajectory plots
    green_patch = Patch(color='darkgreen', label='Logit 0')
    blue_patch = Patch(color='royalblue', label='Logit 1')
    red_patch = Patch(color='darkred', label='Logit 2')
    black_x = plt.Line2D([0], [0], marker='x', color='black', markersize=10, label='Fixed Point')
    
    # Create legend for contour plots
    contour_handles = [plt.Line2D([0], [0], color=contour_colors[i], linestyle='solid') 
                      for i in range(len(variances))]
    contour_labels = [f'Variance={variances[i]}' for i in range(len(variances))]
    
    # Combine all legend elements
    legend_handles = [green_patch, blue_patch, red_patch, black_x] + contour_handles
    legend_labels = ['Logit 0', 'Logit 1', 'Logit 2', 'Fixed Point'] + contour_labels
    
    # Add the legend to the bottom of the figure
    fig.legend(handles=legend_handles, labels=legend_labels, 
              loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=5, fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0.1, 1, 0.95])  # Leave space for the row labels and legend
    
    # Save the figure
    save_path = "combined_grid_plot.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Combined grid plot saved to {save_path}")

if __name__ == "__main__":
    create_combined_grid_plot()

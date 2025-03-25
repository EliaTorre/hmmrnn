import matplotlib.pyplot as plt
import numpy as np

def create_vector_grid(x_range=(-5, 5), y_range=(-5, 5), resolution=1.0):
    x = np.arange(x_range[0], x_range[1] + resolution, resolution)
    y = np.arange(y_range[0], y_range[1] + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    vector_grid = np.stack((X, Y), axis=-1)
    return vector_grid

def normalize_vectors(vectors):
    """Normalize vectors to unit length"""
    magnitudes = np.sqrt(np.sum(vectors**2, axis=2))
    # Avoid division by zero
    magnitudes = np.where(magnitudes == 0, 1e-10, magnitudes)
    normalized = vectors / magnitudes[:, :, np.newaxis]
    return normalized

def direction_grid(vector_grid, direction):
    rows, cols = vector_grid.shape[0], vector_grid.shape[1]
    final_vectors = np.zeros((rows, cols, 2))
    
    # Original directions (uniform magnitude, independent of position)
    if direction == 'up':
        final_vectors[:, :, 0] = 0
        final_vectors[:, :, 1] = 1
    elif direction == 'down':
        final_vectors[:, :, 0] = 0
        final_vectors[:, :, 1] = -1
    elif direction == 'right':
        final_vectors[:, :, 0] = 1
        final_vectors[:, :, 1] = 0
    elif direction == 'left':
        final_vectors[:, :, 0] = -1
        final_vectors[:, :, 1] = 0
    elif direction == 'angle':
        angle_rad = np.deg2rad(275)
        final_vectors[:, :, 0] = np.cos(angle_rad)
        final_vectors[:, :, 1] = np.sin(angle_rad)
    
    # New directions (uniform magnitude, direction determined by position)
    elif direction == 'towards_center':
        # Only use position to determine direction
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                # Calculate direction (not magnitude) from position to center
                if x == 0 and y == 0:  # At center
                    final_vectors[i, j, 0] = 0
                    final_vectors[i, j, 1] = 0
                else:
                    # Direction vector pointing toward center (0,0)
                    magnitude = np.sqrt(x**2 + y**2)
                    final_vectors[i, j, 0] = -x / magnitude
                    final_vectors[i, j, 1] = -y / magnitude
    
    elif direction == 'away_from_center':
        # Define vectors for each quadrant that need to be normalized
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                
                if x < 0 and y > 0:  # Second quadrant: left+up
                    vector = np.array([-1, 1])
                elif x > 0 and y > 0:  # First quadrant: right+up
                    vector = np.array([1, 1])
                elif x > 0 and y < 0:  # Fourth quadrant: right+down
                    vector = np.array([1, -1])
                elif x < 0 and y < 0:  # Third quadrant: left+down
                    vector = np.array([-1, -1])
                elif x == 0 and y > 0:  # Positive y-axis: up
                    vector = np.array([0, 1])
                elif x > 0 and y == 0:  # Positive x-axis: right
                    vector = np.array([1, 0])
                elif x == 0 and y < 0:  # Negative y-axis: down
                    vector = np.array([0, -1])
                elif x < 0 and y == 0:  # Negative x-axis: left
                    vector = np.array([-1, 0])
                else:  # Origin (0,0)
                    vector = np.array([0, 0])
                    
                final_vectors[i, j] = vector
    
    elif direction == 'split_horizontal':
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                if x > 0:
                    final_vectors[i, j, 0] = 1
                    final_vectors[i, j, 1] = 0
                elif x < 0:
                    final_vectors[i, j, 0] = -1
                    final_vectors[i, j, 1] = 0
                else:  # x = 0
                    final_vectors[i, j, 0] = 0
                    final_vectors[i, j, 1] = 0
    
    elif direction == 'split_vertical':
        # Top half points up, bottom half points down (unit vectors)
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                if y > 0:
                    final_vectors[i, j, 0] = 0
                    final_vectors[i, j, 1] = 1
                elif y < 0:
                    final_vectors[i, j, 0] = 0
                    final_vectors[i, j, 1] = -1
                else:  # y = 0
                    final_vectors[i, j, 0] = 0
                    final_vectors[i, j, 1] = 0
    
    elif direction == 'clockwise_circular':
        # Define vectors for each quadrant that need to be normalized
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                
                if x < 0 and y > 0:  # Second quadrant
                    # right+up
                    vector = np.array([1, 1])
                elif x > 0 and y > 0:  # First quadrant
                    # right+down
                    vector = np.array([1, -1])
                elif x > 0 and y < 0:  # Fourth quadrant
                    # left+down
                    vector = np.array([-1, -1])
                elif x < 0 and y < 0:  # Third quadrant
                    # left+up
                    vector = np.array([-1, 1])
                elif x == 0 and y > 0:  # Positive y-axis
                    vector = np.array([1, 0])  # Right
                elif x > 0 and y == 0:  # Positive x-axis
                    vector = np.array([0, -1])  # Down
                elif x == 0 and y < 0:  # Negative y-axis
                    vector = np.array([-1, 0])  # Left
                elif x < 0 and y == 0:  # Negative x-axis
                    vector = np.array([0, 1])  # Up
                else:  # Origin (0,0)
                    vector = np.array([0, 0])

                final_vectors[i, j] = vector
    
    elif direction == 'counter_clockwise_circular':
        # Define vectors for each quadrant that need to be normalized
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                
                if x < 0 and y > 0:  # Second quadrant
                    # For a vector like (-1, 1), (-y, x) becomes (-1, -1): left+down
                    vector = np.array([-1, -1])
                elif x > 0 and y > 0:  # First quadrant
                    # For (1, 1), (-y, x) becomes (-1, 1): left+up
                    vector = np.array([-1, 1])
                elif x > 0 and y < 0:  # Fourth quadrant
                    # For (1, -1), (-y, x) becomes (1, 1): right+up
                    vector = np.array([1, 1])
                elif x < 0 and y < 0:  # Third quadrant
                    # For (-1, -1), (-y, x) becomes (1, -1): right+down
                    vector = np.array([1, -1])
                elif x == 0 and y > 0:  # Positive y-axis (up)
                    # (0, positive) rotates to (-positive, 0): left
                    vector = np.array([-1, 0])
                elif x > 0 and y == 0:  # Positive x-axis (right)
                    # (positive, 0) rotates to (0, positive): up
                    vector = np.array([0, 1])
                elif x == 0 and y < 0:  # Negative y-axis (down)
                    # (0, negative) rotates to (-negative, 0): right
                    vector = np.array([1, 0])
                elif x < 0 and y == 0:  # Negative x-axis (left)
                    # (negative, 0) rotates to (0, -negative): down
                    vector = np.array([0, -1])
                else:  # Origin (0,0)
                    vector = np.array([0, 0])
                    
                final_vectors[i, j] = vector
    
    elif direction == 'vortex':
        # Vortex field - combination of circular and radial (unit vectors)
        for i in range(rows):
            for j in range(cols):
                x, y = vector_grid[i, j]
                if x == 0 and y == 0:  # At center
                    final_vectors[i, j, 0] = 0
                    final_vectors[i, j, 1] = 0
                else:
                    # Circular component (tangent)
                    magnitude = np.sqrt(x**2 + y**2)
                    circular_x = -y / magnitude
                    circular_y = x / magnitude
                    
                    # Radial component (outward)
                    radial_x = x / magnitude
                    radial_y = y / magnitude
                    
                    # Mix them (70% circular, 30% radial)
                    mixed_x = 0.7 * circular_x + 0.3 * radial_x
                    mixed_y = 0.7 * circular_y + 0.3 * radial_y
                    
                    # Normalize final vector
                    mix_magnitude = np.sqrt(mixed_x**2 + mixed_y**2)
                    final_vectors[i, j, 0] = mixed_x / mix_magnitude
                    final_vectors[i, j, 1] = mixed_y / mix_magnitude
    
    return final_vectors

def relu(x):
    result = x.copy()
    result[result < 0] = 0
    return result

def visualize_flow(direction, x_range=(-5, 5), y_range=(-5, 5), resolution=1.0, 
                   title_prefix="Flow Field Visualization", arrow_color='blue'):
    # Create the vector grid internally
    vector_grid = create_vector_grid(x_range, y_range, resolution)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate direction vectors
    direction_vectors = direction_grid(vector_grid, direction)
    
    # Extract coordinates
    X = vector_grid[:, :, 0]
    Y = vector_grid[:, :, 1]
    
    # Plot original vectors on the left subplot
    U1 = direction_vectors[:, :, 0]
    V1 = direction_vectors[:, :, 1]
    q1 = ax1.quiver(X, Y, U1, V1, color=arrow_color, scale_units='xy', scale=2)
    ax1.set_title(f"{direction} - Before ReLU")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    
    # Apply ReLU and plot on the right subplot
    relu_vectors = relu(direction_vectors)
    U2 = relu_vectors[:, :, 0]
    V2 = relu_vectors[:, :, 1]
    q2 = ax2.quiver(X, Y, U2, V2, color=arrow_color, scale_units='xy', scale=2)
    ax2.set_title(f"{direction} - After ReLU")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig, (ax1, ax2)



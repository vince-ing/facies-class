#!/usr/bin/env python

"""

Loads the real 2D seismic grids and the synthetic facies "training data"
to perform a statistical classification.

This script implements Step 6 of the rock physics workflow:
"Step 6: Compute the Mahalanobis Distance for each real data point"

Inputs:
    - intercept_gradient_data.pkl: (From Step 5) A dictionary containing the
                                   Intercept/Gradient clouds for each facies.
    - Seismic/SeismicIntercept.txt: The 2D (n_inline x n_xline) grid of intercept data.
    - Seismic/SeismicGradient.txt: The 2D (n_inline x n_xline) grid of gradient data.
    - Seismic/SeismicInlineNumbers.txt: The 1D array (n_inline) of inline coordinates.
    - Seismic/SeismicXlineNumers.txt: The 1D array (n_xline) of xline coordinates.

Outputs:
    - most_likely_facies_map.png: A 2D map of the most likely facies.
    - grouped_facies_map.png: A simplified map (Oil Sand, Brine Sand, Shale).
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

# --- Constants (Must match previous scripts) ---

# Define the 9 facies cases in a consistent order
FACIES_NAMES = [
    'FaciesIIaOil', 'FaciesIIbOil', 'FaciesIIcOil',
    'FaciesIIa',     'FaciesIIb',     'FaciesIIc',
    'FaciesIII',     'FaciesIV',      'FaciesV'
]

# Map names to their numeric ID (1-9)
FACIES_MAP = {name: i + 1 for i, name in enumerate(FACIES_NAMES)}

# Generates a continuous gradient from black to white
print("Generating 9-step grayscale colormap...")
# Get 9 evenly spaced values between 0.0 (black) and 1.0 (white)
color_values = np.linspace(0, 1, len(FACIES_NAMES))

# Get the 'gray' colormap (0.0=black, 1.0=white)
cmap_gray = cm.get_cmap('gray')

# Build the new FACIES_COLORS dictionary
FACIES_COLORS = {}
for i, name in enumerate(FACIES_NAMES):
    # cmap_gray(value) returns an (R, G, B, A) tuple
    # We convert it to a hex string for consistency
    color_rgba = cmap_gray(color_values[i])
    # Convert (R,G,B) to hex
    FACIES_COLORS[name] = mcolors.to_hex(color_rgba[:3])
    print(f"  {name} -> {FACIES_COLORS[name]}")

# Define the grouped facies mapping (as seen on lecture page 31)
GROUPED_MAP = {
    'FaciesIIaOil': 1, 'FaciesIIbOil': 1, 'FaciesIIcOil': 1,
    'FaciesIIa':     2, 'FaciesIIb':     2, 'FaciesIIc':     2, 'FaciesIII': 2,
    'FaciesIV':      3, 'FaciesV':       3
}
GROUPED_NAMES = ['Oil Sand', 'Brine Sand', 'Shale']
GROUPED_COLORS = ['#202020', '#7a7a7a', '#dddddd'] 


# --- Functions ---

def load_training_data(filename="intercept_gradient_data.pkl"):
    """
    Loads the synthetic Intercept/Gradient data from Step 5.
    """
    print(f"Loading synthetic training data from {filename}...")
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print("Load successful.")
        return data
    except FileNotFoundError:
        print(f"Error: Training data '{filename}' not found.")
        print("Please run 'step5_compute_gradient.py' first.")
        exit(1)

def load_seismic_data(seismic_dir="Seismic"):
    """
    Loads all real seismic data files, assuming a 2D grid structure.
    """
    print(f"Loading real seismic data from {seismic_dir}/...")
    try:
        # File paths
        # Note the typo in "SeismicXlineNumers.txt"
        path_intercept = os.path.join(seismic_dir, 'SeismicIntercept.txt')
        path_gradient = os.path.join(seismic_dir, 'SeismicGradient.txt')
        path_inline = os.path.join(seismic_dir, 'SeismicInlineNumbers.txt')
        path_xline = os.path.join(seismic_dir, 'SeismicXlineNumers.txt')

        # Load coordinate axes first
        inlines = np.loadtxt(path_inline)
        xlines = np.loadtxt(path_xline)
        
        n_rows = len(inlines)
        n_cols = len(xlines)
        
        print(f"  Loaded {n_rows} inline numbers (Y-axis).")
        print(f"  Loaded {n_cols} xline numbers (X-axis).")

        # Now load the 2D data grids
        intercept_grid = np.loadtxt(path_intercept)
        gradient_grid = np.loadtxt(path_gradient)
        
        print(f"  Loaded Intercept grid with shape: {intercept_grid.shape}")
        print(f"  Loaded Gradient grid with shape: {gradient_grid.shape}")

        # --- CRITICAL VALIDATION ---
        expected_shape = (n_rows, n_cols)
        
        if intercept_grid.shape != expected_shape:
            raise ValueError(
                f"Intercept grid shape {intercept_grid.shape} does not match "
                f"coordinate axis lengths {expected_shape}."
            )
        if gradient_grid.shape != expected_shape:
            raise ValueError(
                f"Gradient grid shape {gradient_grid.shape} does not match "
                f"coordinate axis lengths {expected_shape}."
            )

        seismic_data = {
            'intercept_grid': intercept_grid,
            'gradient_grid': gradient_grid,
            'inlines': inlines,
            'xlines': xlines
        }
        
        print("Load successful. All data grids and coordinates are consistent.")
        return seismic_data

    except FileNotFoundError as e:
        print(f"Error: Missing seismic file: {e.filename}")
        print("Please ensure all seismic data files are in the 'Seismic' directory.")
        exit(1)
    except Exception as e:
        print(f"Error loading seismic data: {e}")
        exit(1)

def compute_class_statistics(training_data, facies_names):
    """
    Computes the Mean Vector (mu) and Inverse Covariance Matrix (inv_cov)
    for each facies class. This is the "training" step.
    """
    print("Computing statistics (Mean and Covariance) for all facies classes...")
    class_stats = {}
    for name in facies_names:
        if name not in training_data:
            print(f"Warning: No training data for {name}. Skipping.")
            continue
            
        # Get data for this class
        intercept = training_data[name]['intercept']
        gradient = training_data[name]['gradient']
        
        # Stack into an (N_samples, 2) array
        data_stack = np.stack([intercept, gradient], axis=1)
        
        # mu: Mean vector [mean_intercept, mean_gradient]
        mean_vec = np.mean(data_stack, axis=0)
        
        # cov: 2x2 Covariance matrix
        cov_matrix = np.cov(data_stack.T)
        
        # inv_cov: 2x2 Inverse Covariance Matrix
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print(f"Error: Covariance matrix for {name} is singular. Cannot invert.")
            continue
            
        class_stats[name] = {
            'mean_vec': mean_vec,
            'inv_cov': inv_cov_matrix
        }
    
    print("Class statistics computation complete.")
    return class_stats

def classify_seismic_data(seismic_data, class_stats, facies_names):
    """
    Classifies every seismic data point (pixel) using the Mahalanobis Distance.
    
    D^2 = (x - mu)^T * inv_cov * (x - mu)
    
    This function is vectorized for speed.
    """
    print("Classifying all seismic data points...")
    intercept_grid = seismic_data['intercept_grid']
    gradient_grid = seismic_data['gradient_grid']
    n_rows, n_cols = intercept_grid.shape
    n_pixels = n_rows * n_cols
    
    # Stack real data into a (N_pixels, 2) array
    # [ [intercept_1, gradient_1],
    #   [intercept_2, gradient_2], ... ]
    all_pixels = np.stack([intercept_grid.ravel(), gradient_grid.ravel()], axis=-1)
    
    # Store distances for each class in an (N_pixels, N_classes) array
    n_classes = len(facies_names)
    all_distances = np.zeros((n_pixels, n_classes))
    
    for i, name in enumerate(facies_names):
        if name not in class_stats:
            all_distances[:, i] = np.inf # Penalize if class has no stats
            continue
            
        stats = class_stats[name]
        mean_vec = stats['mean_vec']
        inv_cov = stats['inv_cov']
        
        # (x - mu)
        diff = all_pixels - mean_vec
        
        # (x - mu) * inv_cov
        temp = np.dot(diff, inv_cov)
        
        # (x - mu)^T * inv_cov * (x - mu)
        # We get D^2 for all pixels by summing the element-wise product
        d_squared = np.sum(temp * diff, axis=1)
        
        all_distances[:, i] = d_squared
        
    # Find the winner (minimum D^2) for each pixel
    # winner_indices is a 1D array (N_pixels) with values from 0-8
    winner_indices = np.argmin(all_distances, axis=1)
    
    # Map from 0-8 index to 1-9 facies ID
    facies_id_map = {i: FACIES_MAP[name] for i, name in enumerate(facies_names)}
    classification_flat = np.array([facies_id_map[idx] for idx in winner_indices])
    
    # Reshape back to the 2D grid
    classification_grid = classification_flat.reshape(n_rows, n_cols)
    
    print("Classification complete.")
    return classification_grid

def plot_classification_map(grid, seismic_data, title, filename, 
                            facies_names, colors, id_map):
    """
    Generic plotting function for generating the final facies maps.
    """
    print(f"Generating plot: {title}...")
    
    inlines = seismic_data['inlines']
    xlines = seismic_data['xlines']
    
    # Create the custom colormap
    color_list = [colors[name] for name in facies_names]
    cmap = mcolors.ListedColormap(color_list)
    
    # Create a norm to map discrete facies IDs
    boundaries = sorted(id_map.values())
    # Create boundaries centered on the integer values
    b_all = [b - 0.5 for b in boundaries]
    b_all.append(boundaries[-1] + 0.5)
    norm = mcolors.BoundaryNorm(b_all, cmap.N)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto',
                   origin='lower',
                   extent=[xlines.min(), xlines.max(),
                           inlines.min(), inlines.max()])
    
    ax.set_xlabel('Xline', fontsize=12)
    ax.set_ylabel('Inline', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add Colorbar
    # Ticks should be at the center of the color block (the integer ID)
    cbar_ticks = sorted(id_map.values())
    cbar = fig.colorbar(im, ticks=cbar_ticks)
    cbar.set_ticklabels(facies_names)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved map to {filename}")
    plt.close(fig)

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Load synthetic "training" data (from Step 5)
    training_data = load_training_data("intercept_gradient_data.pkl")
    
    # 2. Load real seismic data (2D grid version)
    seismic_data = load_seismic_data("Seismic")
    
    # 3. Compute class statistics (mu, inv_cov)
    class_stats = compute_class_statistics(training_data, FACIES_NAMES)
    
    # 4. Classify seismic data
    classification_grid = classify_seismic_data(seismic_data, class_stats, FACIES_NAMES)
    
    # 5. Plot the "Most Likely Facies" map (9 facies)
    plot_classification_map(
        grid=classification_grid,
        seismic_data=seismic_data,
        title="Most Likely Facies",
        filename="most_likely_facies_map.png",
        facies_names=FACIES_NAMES,
        colors=FACIES_COLORS,
        id_map=FACIES_MAP
    )
    
    # 6. Create the "Grouped Facies" map
    # Vectorized way to map 1-9 IDs to 1-3 IDs
    map_1_9_to_1_3 = {FACIES_MAP[name]: GROUPED_MAP[name] for name in FACIES_NAMES}
    grouped_grid = np.vectorize(map_1_9_to_1_3.get)(classification_grid)
    
    # Create grouped colors/names/id map
    grouped_colors_dict = {name: GROUPED_COLORS[i] for i, name in enumerate(GROUPED_NAMES)}
    grouped_id_map = {name: i + 1 for i, name in enumerate(GROUPED_NAMES)}

    plot_classification_map(
        grid=grouped_grid,
        seismic_data=seismic_data,
        title="Most Likely Grouped Facies",
        filename="grouped_facies_map.png",
        facies_names=GROUPED_NAMES,
        colors=grouped_colors_dict,
        id_map=grouped_id_map
    )
    
    print("\nStep 6 complete. Workflow finished.")
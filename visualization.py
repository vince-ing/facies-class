import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Import the data loading function from your other file
# This assumes 'load_data.py' is in the same directory
try:
    from load_data import load_data
except ImportError:
    print("Error: Could not import 'load_data.py'.")
    print("Please make sure 'load_data.py' is in the same directory as 'visualization.py'.")
    exit()

def create_blocked_gray_cmap():
    """
    Translates the BlockedGray.m logic into a matplotlib colormap.
    
    We will save this function for the final facies plot.
    
    Returns:
        matplotlib.colors.ListedColormap: The custom 'blocked_gray' colormap.
    """
    print("Creating 'blocked_gray' colormap...")
    
    # 1. Get the standard 'gray' colormap
    base_cmap = plt.get_cmap('gray')
    
    # 2. Get the three specific colors, normalized from 0.0 to 1.0
    #    (MATLAB 1/8, 1/2, 7/8)
    color_dark = base_cmap(1/8)
    color_mid = base_cmap(1/2)
    color_light = base_cmap(7/8)
    
    # 3. Build the new 64-entry color list
    #    (21 dark, 21 mid, 22 light)
    new_colors = ([color_dark] * 21) + ([color_mid] * 21) + ([color_light] * 22)
    
    # 4. Create the new ListedColormap
    blocked_gray_cmap = ListedColormap(new_colors, name='blocked_gray')
    
    return blocked_gray_cmap

def plot_seismic_data(seismic_data, cmap):
    """
    Plots the seismic intercept and gradient data.
    
    Args:
        seismic_data (dict): The dictionary containing seismic data 
                             (e.g., all_data['seismic']).
        cmap (str or matplotlib.colors.ListedColormap): The colormap to use.
    """
    print("Plotting seismic data...")
    
    # Get the data to plot
    intercept_data = seismic_data['intercept']
    gradient_data = seismic_data['gradient']
    
    # Get the axis vectors
    xline_vec = seismic_data['xline']
    inline_vec = seismic_data['inline']
    
    # Define the plot extent [x_min, x_max, y_min, y_max]
    plot_extent = [
        xline_vec.min(), xline_vec.max(),
        inline_vec.min(), inline_vec.max()
    ]
    
    # ==================
    # *** CHANGES ***
    # ==================
    
    # Create the figure and subplots
    # Removed 'sharey=True' so both plots get Y-axis labels
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # --- Plot 1: Intercept ---
    ax0 = axes[0]
    im0 = ax0.imshow(
        intercept_data, 
        cmap=cmap, 
        extent=plot_extent,
        aspect='equal',
        origin='lower'
    )
    # Added 'shrink=0.75' to make the colorbar smaller
    fig.colorbar(im0, ax=ax0, label='Intercept Value', shrink=0.75)
    ax0.set_title('Seismic Intercept', fontsize=16)
    ax0.set_xlabel('X-Line', fontsize=12)
    ax0.set_ylabel('In-Line', fontsize=12)
    
    # --- Plot 2: Gradient ---
    ax1 = axes[1]
    im1 = ax1.imshow(
        gradient_data, 
        cmap=cmap, 
        extent=plot_extent,
        aspect='equal',
        origin='lower'
    )
    # Added 'shrink=0.75' to make the colorbar smaller
    fig.colorbar(im1, ax=ax1, label='Gradient Value', shrink=0.75)
    ax1.set_title('Seismic Gradient', fontsize=16)
    ax1.set_xlabel('X-Line', fontsize=12)
    ax1.set_ylabel('In-Line', fontsize=12) # This label will now appear
    
    # --- Final Touches ---
    fig.suptitle('Seismic Intercept and Gradient Surfaces', fontsize=20, y=1.02)
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'seismic_data_plots.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nSuccessfully saved plot to: {output_filename}")
    
    # Optionally, display the plot
    # plt.show()

# --- Main execution ---
# This ensures the code only runs when you execute 'visualization.py' directly
if __name__ == "__main__":
    
    # 1. Load all data from your other script
    print("Loading data via load_data.py...")
    all_data = load_data()
    
    # 2. Define the colormap for this plot
    cmap_to_use = 'gray_r' 
    
    # 3. Check if seismic data was loaded successfully
    if 'seismic' in all_data and 'intercept' in all_data['seismic']:
        # 4. Plot the data
        print(f"Using colormap: '{cmap_to_use}'")
        plot_seismic_data(all_data['seismic'], cmap_to_use)
    else:
        print("\nError: Seismic data not found in 'all_data' dictionary.")
        print("Cannot generate plots.")
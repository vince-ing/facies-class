import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pickle
from scipy.stats import norm

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
    
    # Create the figure and subplots
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
    fig.colorbar(im1, ax=ax1, label='Gradient Value', shrink=0.75)
    ax1.set_title('Seismic Gradient', fontsize=16)
    ax1.set_xlabel('X-Line', fontsize=12)
    ax1.set_ylabel('In-Line', fontsize=12) # This label will now appear
    
    # --- Final Touches ---
    fig.suptitle('Seismic Intercept and Gradient Surfaces', fontsize=20, y=1.02)
    
    # Use the object-oriented method
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the figure
    output_filename = 'seismic_data_plots.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nSuccessfully saved plot to: {output_filename}")
    
    # Explicitly close the figure to free memory
    plt.close(fig)

def load_statistics(filename='facies_statistics.pkl'):
    """
    Loads the pickled statistics file.
    """
    print(f"Loading statistics from: {filename}...")
    try:
        with open(filename, 'rb') as f:
            stats_data = pickle.load(f)
        print("  Statistics loaded successfully.")
        
        # DEBUG: Print what we actually loaded
        print("\n=== DEBUG: Statistics Content ===")
        print(f"Keys in stats_data: {stats_data.keys()}")
        if 'plot_ranges' in stats_data:
            print(f"\nPlot ranges available for: {stats_data['plot_ranges'].keys()}")
            for prop, ranges in stats_data['plot_ranges'].items():
                print(f"  {prop}: min={ranges['min']:.4f}, max={ranges['max']:.4f}")
        
        if 'facies_stats' in stats_data:
            print(f"\nFacies available: {list(stats_data['facies_stats'].keys())}")
            # Print sample stats for first facies
            first_facies = list(stats_data['facies_stats'].keys())[0]
            print(f"\nSample stats for {first_facies}:")
            for prop in ['Ip', 'VpVs']:
                if prop in stats_data['facies_stats'][first_facies]['univariate']:
                    s = stats_data['facies_stats'][first_facies]['univariate'][prop]
                    print(f"  {prop}: mean={s['mean']:.4f}, std={s['std']:.4f}")
        print("=================================\n")
        
        return stats_data
    except FileNotFoundError:
        print(f"  Error: Statistics file '{filename}' not found.")
        print("  Please run 'statistics.py' first.")
        return None
    except Exception as e:
        print(f"  Error loading {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_pdf_cdf(stats_data, facies_to_plot, colors):
    """
    Plots the PDF and CDF curves for selected facies and properties.
    """
    if not stats_data:
        print("No statistics data provided. Cannot plot PDFs/CDFs.")
        return
    
    # Validate that we have the required data
    if 'plot_ranges' not in stats_data:
        print("ERROR: 'plot_ranges' not found in statistics data!")
        return
    
    if 'facies_stats' not in stats_data:
        print("ERROR: 'facies_stats' not found in statistics data!")
        return
        
    print("Generating PDF/CDF plots...")

    properties = {
        'Ip': 'AI',
        'VpVs': 'Vp/Vs'
    }

    # Check if the properties exist in plot_ranges
    for prop_key in properties.keys():
        if prop_key not in stats_data['plot_ranges']:
            print(f"ERROR: '{prop_key}' not found in plot_ranges!")
            return

    # === 1. Create PDF PLOT ===
    fig_pdf, axes_pdf = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    
    for ax, (prop_key, x_label) in zip(axes_pdf, properties.items()):
        plot_range = stats_data['plot_ranges'][prop_key]
        
        print(f"  Plotting {prop_key}: range [{plot_range['min']:.4f}, {plot_range['max']:.4f}]")
        
        # Create x values with some padding
        range_width = plot_range['max'] - plot_range['min']
        x_min = plot_range['min'] - 0.1 * range_width
        x_max = plot_range['max'] + 0.1 * range_width
        x = np.linspace(x_min, x_max, 400)
        
        plot_count = 0
        for facies_name in facies_to_plot:
            if facies_name not in stats_data['facies_stats']:
                print(f"  Warning: '{facies_name}' not found in stats. Skipping.")
                continue
            
            if prop_key not in stats_data['facies_stats'][facies_name]['univariate']:
                print(f"  Warning: '{prop_key}' not found for '{facies_name}'. Skipping.")
                continue
            
            stats = stats_data['facies_stats'][facies_name]['univariate'][prop_key]
            mean = stats['mean']
            std = stats['std']
            
            print(f"    {facies_name}: mean={mean:.4f}, std={std:.4f}")
            
            y_pdf = norm.pdf(x, mean, std)
            ax.plot(x, y_pdf, label=facies_name, color=colors[facies_name], linewidth=2)
            plot_count += 1
        
        if plot_count == 0:
            print(f"  WARNING: No data was plotted for {prop_key}!")
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'PDF for {prop_key}', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig_pdf.suptitle('Probability Distribution Functions (PDFs)', fontsize=18, y=1.02)
    fig_pdf.tight_layout(rect=[0, 0, 1, 0.98])
    
    pdf_filename = 'facies_pdf_plots.png'
    plt.savefig(pdf_filename, bbox_inches='tight', dpi=150)
    print(f"  Saved PDF plot to: {pdf_filename}")
    plt.close(fig_pdf)

    # === 2. Create CDF PLOT ===
    fig_cdf, axes_cdf = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    for ax, (prop_key, x_label) in zip(axes_cdf, properties.items()):
        plot_range = stats_data['plot_ranges'][prop_key]
        
        range_width = plot_range['max'] - plot_range['min']
        x_min = plot_range['min'] - 0.1 * range_width
        x_max = plot_range['max'] + 0.1 * range_width
        x = np.linspace(x_min, x_max, 400)
        
        for facies_name in facies_to_plot:
            if facies_name not in stats_data['facies_stats']:
                continue
            
            if prop_key not in stats_data['facies_stats'][facies_name]['univariate']:
                continue
            
            stats = stats_data['facies_stats'][facies_name]['univariate'][prop_key]
            mean = stats['mean']
            std = stats['std']
            
            y_cdf = norm.cdf(x, mean, std)
            ax.plot(x, y_cdf, label=facies_name, color=colors[facies_name], linewidth=2)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title(f'CDF for {prop_key}', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig_cdf.suptitle('Cumulative Distribution Functions (CDFs)', fontsize=18, y=1.02)
    fig_cdf.tight_layout(rect=[0, 0, 1, 0.98])
    
    cdf_filename = 'facies_cdf_plots.png'
    plt.savefig(cdf_filename, bbox_inches='tight', dpi=150)
    print(f"  Saved CDF plot to: {cdf_filename}")
    plt.close(fig_cdf)
    

# --- Main execution ---
if __name__ == "__main__":
    
    # 1. Load data
    print("Loading data via load_data.py...")
    all_data = load_data()
    
    # 2. Plot seismic data
    cmap_to_use = 'gray_r' 
    if 'seismic' in all_data and 'intercept' in all_data['seismic']:
        print(f"\nUsing colormap: '{cmap_to_use}'")
        plot_seismic_data(all_data['seismic'], cmap_to_use)
    else:
        print("\nError: Seismic data not found. Cannot generate seismic plots.")
        
    
    # === PDF/CDF PLOTTING ===
    print("\n--- Generating PDF/CDF Plots ---")
    
    # 1. Load statistics
    stats_file = 'facies_statistics.pkl'
    all_stats = load_statistics(stats_file)
    
    if all_stats:
        # 2. Define facies and colors
        facies_list = [
            'FaciesIIa', 
            'FaciesIIb', 
            'FaciesIIc', 
            'FaciesIII', 
            'FaciesIV', 
            'FaciesV'
        ]
        
        facies_colors = {
            'FaciesIIa': 'brown',  # Brown
            'FaciesIIb': 'orange',  # Orange
            'FaciesIIc': 'm',  # Magenta
            'FaciesIII': 'g',  # Green
            'FaciesIV': 'c',   # Cyan
            'FaciesV': 'b'     # Blue
        }
        
        # 3. Call the plotting function
        plot_pdf_cdf(all_stats, facies_list, facies_colors)
    else:
        print("\nError: Statistics data not found. Skipping PDF/CDF plots.")
        print("\nPlease run 'statistics.py' first to generate the statistics file!")
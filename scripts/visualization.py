"""
scripts/visualization.py

This module is the dedicated "artist" for the pipeline.
It contains all functions for generating and saving plots.
It is a "worker" module that takes data and configuration
as arguments and does not read or write files itself,
other than saving the final plot images.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from scipy.stats import norm

# ==============================================================================
#   Module-Level Styling Constants
# ==============================================================================

# Facies colors from the original project for consistency
FACIES_COLORS = {
    'FaciesIIa': '#8B0000',     # dark red
    'FaciesIIaOil': '#FF0000',  # red
    'FaciesIIb': '#FFA500',     # orange
    'FaciesIIbOil': '#FFD700',  # gold
    'FaciesIIc': '#FF00FF',     # magenta
    'FaciesIIcOil': '#EE82EE',  # violet
    'FaciesIII': '#008000',    # green
    'FaciesIV': '#00FFFF',     # cyan
    'FaciesV': '#0000FF',      # blue
}

# Grouped facies definitions (Visualization-only logic)
GROUPED_FACIES_MAP = {
    'FaciesIIa': 1,     # Sand
    'FaciesIIaOil': 1,  # Sand
    'FaciesIIb': 1,     # Sand
    'FaciesIIbOil': 1,  # Sand
    'FaciesIIc': 1,     # Sand
    'FaciesIIcOil': 1,  # Sand
    'FaciesIII': 2,     # Shaly Sand
    'FaciesIV': 3,      # Shale
    'FaciesV': 3,       # Shale
}

GROUPED_FACIES_NAMES = ['Sand', 'Shaly Sand', 'Shale']
GROUPED_FACIES_CMAP = ListedColormap(['#FFFF00', '#808080', '#008000']) # Yellow, Gray, Green


# ==============================================================================
#   a) Well Log Plot
# ==============================================================================

def plot_well_logs(well_data, output_path):
    """
    Plots the Well2.txt log data (Vp, Vs, Density, GR, Porosity).
    """
    print(f"Generating well log plot: {output_path}")
    
    try:
        logs_to_plot = ['Vp', 'Vs', 'Density', 'GR', 'Porosity']
        log_colors = ['blue', 'red', 'green', 'black', 'purple']
        log_units = ['(km/s)', '(km/s)', '(g/cc)', '(API)', '(v/v)']
        
        depth = well_data['Depth']
        
        fig, axes = plt.subplots(1, len(logs_to_plot), figsize=(12, 10), sharey=True)
        fig.suptitle('Well 2 Log Data', fontsize=16)
        
        for i, (log_name, color, unit) in enumerate(zip(logs_to_plot, log_colors, log_units)):
            ax = axes[i]
            if log_name in well_data:
                log_data = well_data[log_name]
                ax.plot(log_data, depth, color=color, lw=1)
                ax.set_title(f"{log_name}\n{unit}")
                ax.set_xlabel(None) # Clear x-label
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # --- FIX: Corrected method to move ticks to top ---
                # Position x-axis ticks and labels at the top
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top') 
                ax.tick_params(axis='x', labelsize=8)
                # --- End Fix ---
                
            else:
                ax.set_title(f"{log_name}\n(Not Found)")
                # --- FIX: Also apply tick fix to empty plots ---
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                # --- End Fix ---


        # Set Y-axis (Depth)
        axes[0].set_ylabel('Depth (m)')
        axes[0].invert_yaxis()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"  Error plotting well logs: {e}")

# ==============================================================================
#   b) PDF and CDF Plots
# ==============================================================================

def plot_property_distributions(facies_data_map, facies_names, output_pdf_path, output_cdf_path):
    """
    Generates two plots:
    1. PDFs for 5 key rock properties, with all 9 facies overlaid.
    2. CDFs for the same.
    """
    print(f"Generating PDF/CDF plots...")
    
    properties = ['Vp', 'Vs', 'Density', 'Ip', 'VpVs']
    n_props = len(properties)
    
    fig_pdf, axes_pdf = plt.subplots(n_props, 1, figsize=(10, 12), sharex=False)
    fig_cdf, axes_cdf = plt.subplots(n_props, 1, figsize=(10, 12), sharex=False)
    
    fig_pdf.suptitle('Facies Property - Probability Density Functions (PDF)', fontsize=16)
    fig_cdf.suptitle('Facies Property - Cumulative Density Functions (CDF)', fontsize=16)
    
    for i, prop_name in enumerate(properties):
        ax_pdf = axes_pdf[i]
        ax_cdf = axes_cdf[i]
        
        # Find global min/max for this property to set plot range
        all_vals = []
        for name in facies_names:
            if name in facies_data_map:
                vals = getattr(facies_data_map[name], prop_name, np.array([]))
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    all_vals.append(vals)
        
        if not all_vals:
            ax_pdf.set_title(prop_name)
            ax_cdf.set_title(prop_name)
            continue
            
        full_data = np.concatenate(all_vals)
        p1 = np.percentile(full_data, 1)
        p99 = np.percentile(full_data, 99)
        x_axis = np.linspace(p1, p99, 500)
        
        for name in facies_names:
            if name not in facies_data_map:
                continue
            
            data = getattr(facies_data_map[name], prop_name, np.array([]))
            data = data[np.isfinite(data)]
            
            if data.size < 2:
                continue
                
            mean = np.mean(data)
            std = np.std(data)
            color = FACIES_COLORS.get(name, 'gray')
            
            # PDF
            pdf = norm.pdf(x_axis, mean, std)
            ax_pdf.plot(x_axis, pdf, color=color, label=name)
            
            # CDF
            cdf = norm.cdf(x_axis, mean, std)
            ax_cdf.plot(x_axis, cdf, color=color, label=name)

        ax_pdf.set_title(prop_name)
        ax_pdf.set_ylabel('Probability Density')
        ax_pdf.grid(True, linestyle=':', alpha=0.6)
        
        ax_cdf.set_title(prop_name)
        ax_cdf.set_ylabel('Cumulative Probability')
        ax_cdf.grid(True, linestyle=':', alpha=0.6)

    axes_pdf[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
    axes_cdf[0].legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
    
    fig_pdf.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    fig_cdf.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    
    fig_pdf.savefig(output_pdf_path)
    fig_cdf.savefig(output_cdf_path)
    
    plt.close(fig_pdf)
    plt.close(fig_cdf)
    print(f"  Saved: {output_pdf_path}")
    print(f"  Saved: {output_cdf_path}")

# ==============================================================================
#   c) AVO Reflectivity Curves Plot
# ==============================================================================

def plot_reflectivity_curves(avo_data, facies_names, output_path, n_curves_to_plot=500):
    """
    Plots the synthetic AVO reflectivity curves (Rpp) for all 9 facies
    on a 3x3 grid.
    """
    print(f"Generating AVO reflectivity curves plot: {output_path}")
    
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
        axes = axes.ravel()
        fig.suptitle('AVO Reflectivity (Rpp) vs. Angle', fontsize=18)

        for i, facies_name in enumerate(facies_names):
            ax = axes[i]
            if facies_name not in avo_data:
                ax.set_title(f"{facies_name}\n(No Data)")
                continue

            data = avo_data[facies_name]
            curves = data['Rpp']
            angles = data['angles']
            
            n_simulations = curves.shape[0]
            if n_simulations == 0:
                ax.set_title(f"{facies_name}\n(No Curves)")
                continue

            # Plot a random subset of curves for efficiency
            n_to_plot = min(n_simulations, n_curves_to_plot)
            indices = np.random.choice(n_simulations, n_to_plot, replace=False)
            
            color = FACIES_COLORS.get(facies_name, 'gray')
            
            # Plot subset
            for idx in indices:
                ax.plot(angles, curves[idx, :], color=color, alpha=0.05)
                
            # Plot mean curve
            mean_curve = np.mean(curves, axis=0)
            ax.plot(angles, mean_curve, color='black', lw=2, label='Mean')

            ax.set_title(facies_name)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            if i // 3 == 2: # Bottom row
                ax.set_xlabel('Incidence Angle (degrees)')
            if i % 3 == 0: # Left column
                ax.set_ylabel('Rpp')
                
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        print(f"  Error plotting AVO curves: {e}")

# ==============================================================================
#   d) Intercept-Gradient Crossplots
# ==============================================================================

def plot_intercept_gradient(ig_data, seismic_features, facies_names, output_subplots, output_combined):
    """
    Generates two I-G plots:
    1. A 3x3 grid of I-G point clouds for each synthetic facies.
    2. A combined plot showing real seismic data (hist2d) with synthetic
       facies means overlaid.
    """
    print(f"Generating Intercept-Gradient plots...")
    
    # --- Plot 1: 3x3 Subplots ---
    try:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
        axes = axes.ravel()
        fig.suptitle('Synthetic Intercept vs. Gradient Clouds (by Facies)', fontsize=18)
        
        plot_xlim = (-0.25, 0.25)
        plot_ylim = (-0.4, 0.3)

        for i, facies_name in enumerate(facies_names):
            ax = axes[i]
            if facies_name not in ig_data:
                ax.set_title(f"{facies_name}\n(No Data)")
                continue
                
            data = ig_data[facies_name]
            intercept = data['intercept']
            gradient = data['gradient']
            
            # Clean NaNs
            mask = np.isfinite(intercept) & np.isfinite(gradient)
            intercept = intercept[mask]
            gradient = gradient[mask]
            
            if intercept.size == 0:
                ax.set_title(f"{facies_name}\n(No Valid Data)")
                continue

            color = FACIES_COLORS.get(facies_name, 'gray')
            
            # Plot the point cloud
            ax.scatter(intercept, gradient, c=color, alpha=0.1, s=5)
            
            # Plot the mean value
            mean_intercept = np.mean(intercept)
            mean_gradient = np.mean(gradient)
            ax.scatter(mean_intercept, mean_gradient, c='black', s=50, 
                       edgecolor='white', zorder=5)

            ax.set_title(facies_name)
            ax.set_xlim(plot_xlim)
            ax.set_ylim(plot_ylim)
            ax.grid(True, linestyle=':', alpha=0.7)
            if i // 3 == 2: ax.set_xlabel('Intercept')
            if i % 3 == 0: ax.set_ylabel('Gradient')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_subplots)
        plt.close(fig)
        print(f"  Saved: {output_subplots}")

    except Exception as e:
        print(f"  Error plotting I-G subplots: {e}")

    # --- Plot 2: Combined Plot with Real Data ---
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title('Real Seismic Data vs. Synthetic Facies Means', fontsize=16)

        # Plot seismic data as a 2D histogram (fast and informative)
        # Clean NaNs from seismic features
        mask_seismic = np.all(np.isfinite(seismic_features), axis=1)
        seismic_I = seismic_features[mask_seismic, 0]
        seismic_G = seismic_features[mask_seismic, 1]
        
        ax.hist2d(seismic_I, seismic_G, 
                  bins=100, 
                  cmap='Greys', 
                  norm=LogNorm(vmin=1, vmax=1e4), # Use LogNorm to see data
                  # label='Seismic Data' # Removed as requested
                  )
        
        # Overlay the means of the synthetic facies data
        for facies_name in facies_names:
            if facies_name not in ig_data: continue
            
            data = ig_data[facies_name]
            intercept = data['intercept']
            gradient = data['gradient']
            mask = np.isfinite(intercept) & np.isfinite(gradient)
            
            if mask.sum() == 0: continue

            mean_intercept = np.mean(intercept[mask])
            mean_gradient = np.mean(gradient[mask])
            color = FACIES_COLORS.get(facies_name, 'gray')
            
            ax.scatter(mean_intercept, mean_gradient, 
                       c=color, s=100, edgecolor='black', 
                       label=facies_name, # This label is for the legend
                       zorder=5)

        ax.set_xlabel('Intercept', fontsize=14)
        ax.set_ylabel('Gradient', fontsize=14)
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # --- FIX: Removed legend call to fix UserWarning ---
        # ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
        # --- End Fix ---
        
        # Create a legend manually just for the scatter points
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=FACIES_COLORS.get(name, 'gray'),
                                 edgecolor='black',
                                 label=name) for name in facies_names if name in ig_data]
        ax.legend(handles=legend_elements, loc='center left', 
                  bbox_to_anchor=(1.05, 0.5), fontsize='small')
        
        plt.tight_layout(rect=[0, 0.03, 0.8, 0.95])
        plt.savefig(output_combined)
        plt.close(fig)
        print(f"  Saved: {output_combined}")
        
    except Exception as e:
        print(f"  Error plotting combined I-G plot: {e}")

# ==============================================================================
#   e) & f) Facies Classification Maps
# ==============================================================================

def plot_facies_maps(classification_results, seismic_geometry, facies_names, output_most_likely, output_grouped):
    """
    Generates two plots:
    1. A map of the most-likely facies for each seismic data point.
    2. A map of the "grouped" facies (Sand, Shaly Sand, Shale).
    
    This function correctly uses the original seismic geometry.
    """
    print(f"Generating facies classification maps...")
    
    try:
        # --- 1. Get Data and Geometry ---
        
        # Get the 2D shape of the original seismic data
        shape_2d = seismic_geometry['intercept'].shape
        
        # Get the axis labels
        inlines = seismic_geometry['inline']
        xlines = seismic_geometry['xline']
        
        # Define the map extent for imshow
        # [x_min, x_max, y_max, y_min]
        extent = [xlines.min(), xlines.max(), inlines.max(), inlines.min()]
        
        # Get the 1D array of classification indices
        most_likely_idx_1d = classification_results['most_likely_facies_idx']
        
        # Reshape the 1D result back to the 2D seismic geometry
        facies_map_2d = most_likely_idx_1d.reshape(shape_2d)

        # --- 2. Plot (e) - Most Likely Facies Map ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('Most Likely Facies')
        
        # Create a 9-color colormap
        cmap_9 = plt.cm.get_cmap('jet', 9)
        
        # Plot the map
        cax = ax.imshow(facies_map_2d, 
                        cmap=cmap_9, 
                        aspect='auto', # Use 'auto' with extent
                        extent=extent,
                        vmin=-0.5, 
                        vmax=8.5)
        
        # Set "equal" aspect *after* plotting
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = fig.colorbar(cax, ticks=np.arange(9))
        cbar.ax.set_yticklabels(facies_names)
        
        ax.set_xlabel('X-Line')
        ax.set_ylabel('Inline')
        
        plt.tight_layout()
        plt.savefig(output_most_likely)
        plt.close(fig)
        print(f"  Saved: {output_most_likely}")

        # --- 3. Plot (f) - Grouped Facies Map ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title('Grouped Facies')
        
        # Create the grouped facies map
        # Vectorized mapping is faster than looping
        # Create a "mapper" array
        mapper = np.zeros(max(GROUPED_FACIES_MAP.values()) + 1, dtype=int)
        name_to_idx_map = {name: i for i, name in enumerate(facies_names)}
        
        group_mapper = np.zeros(len(facies_names), dtype=int)
        for name, group_idx in GROUPED_FACIES_MAP.items():
            if name in name_to_idx_map:
                group_mapper[name_to_idx_map[name]] = group_idx
        
        # Apply the mapping
        grouped_facies_map_2d = group_mapper[facies_map_2d]

        # Plot the grouped map
        cax = ax.imshow(grouped_facies_map_2d,
                        cmap=GROUPED_FACIES_CMAP,
                        aspect='auto',
                        extent=extent,
                        vmin=0.5, # Tick 1
                        vmax=3.5) # Tick 3
        
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = fig.colorbar(cax, ticks=[1, 2, 3])
        cbar.ax.set_yticklabels(GROUPED_FACIES_NAMES)
        
        ax.set_xlabel('X-Line')
        ax.set_ylabel('Inline')
        
        plt.tight_layout()
        plt.savefig(output_grouped)
        plt.close(fig)
        print(f"  Saved: {output_grouped}")

    except Exception as e:
        print(f"  Error plotting facies maps: {e}")
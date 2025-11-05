import os
import numpy as np
import pickle
from load_data import load_data

def calculate_statistics(all_data):
    """
    Calculates all required statistics for the 9 facies classes.
    
    This computes:
    1.  Univariate stats (mean, std) for Vp, Vs, Density, Ip, and VpVs.
    2.  Bivariate stats (mean vector, covariance matrix) for [Vp, Vs].
    3.  Global min/max plot ranges for the 5 univariate properties.

    Args:
        all_data (dict): The main data dictionary loaded from load_data.py.

    Returns:
        dict: A nested dictionary containing all statistics.
    """
    
    # This will hold the stats for all facies
    stats_data = {
        'facies_stats': {},
        'plot_ranges': {}
    }
    
    # This will temporarily hold all values to find the global min/max
    plot_range_values = {
        'Vp': [],
        'Vs': [],
        'Density': [],
        'Ip': [],
        'VpVs': []
    }
    
    print("Calculating statistics for each facies...")

    # Loop through each facies in the loaded data
    for facies_name, facies_obj in all_data['facies'].items():
        
        print(f"\n  Processing: {facies_name}")
        
        # This dict will hold stats for this *one* facies
        current_facies_stats = {
            'univariate': {},
            'bivariate_VpVs': {}
        }
        
        # --- 1. Get Property Arrays ---
        # Get the 5 properties required for the PDF plots
        properties = {
            'Vp': facies_obj.Vp,
            'Vs': facies_obj.Vs,
            'Density': facies_obj.Density,
            'Ip': facies_obj.Ip,
            'VpVs': facies_obj.VpVs
        }

        # --- DEBUG: Check for NaN or invalid values ---
        for prop_name, prop_data in properties.items():
            nan_count = np.sum(np.isnan(prop_data))
            inf_count = np.sum(np.isinf(prop_data))
            if nan_count > 0 or inf_count > 0:
                print(f"    WARNING: {prop_name} has {nan_count} NaN and {inf_count} Inf values")
                print(f"    Sample values: {prop_data[:5]}")

        # --- 2. Calculate Univariate Stats ---
        for prop_name, prop_data in properties.items():
            # Remove NaN and Inf values before calculating statistics
            clean_data = prop_data[np.isfinite(prop_data)]
            
            if len(clean_data) == 0:
                print(f"    ERROR: No valid data for {prop_name}!")
                mean = np.nan
                std = np.nan
            else:
                # Calculate mean and standard deviation
                mean = np.mean(clean_data)
                std = np.std(clean_data)
            
            # Store the stats
            current_facies_stats['univariate'][prop_name] = {
                'mean': mean,
                'std': std
            }
            
            # Add all CLEAN data points to our list for finding global min/max
            if len(clean_data) > 0:
                plot_range_values[prop_name].append(clean_data)
            
            print(f"    {prop_name}: mean={mean:.4f}, std={std:.4f} (n={len(clean_data)})")

        # --- 3. Calculate Bivariate Stats for [Vp, Vs] ---
        # Clean both Vp and Vs data
        vp_clean = properties['Vp'][np.isfinite(properties['Vp'])]
        vs_clean = properties['Vs'][np.isfinite(properties['Vs'])]
        
        # Make sure they have the same length
        min_len = min(len(vp_clean), len(vs_clean))
        vp_clean = vp_clean[:min_len]
        vs_clean = vs_clean[:min_len]
        
        if min_len > 0:
            # Create a 2xN array
            vp_vs_stack = np.stack((vp_clean, vs_clean), axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(vp_vs_stack)
            
            # Calculate mean vector
            mean_vector = np.array([
                current_facies_stats['univariate']['Vp']['mean'],
                current_facies_stats['univariate']['Vs']['mean']
            ])
        else:
            print(f"    ERROR: No valid Vp/Vs data for bivariate stats!")
            cov_matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            mean_vector = np.array([np.nan, np.nan])
        
        # Store the bivariate stats
        current_facies_stats['bivariate_VpVs'] = {
            'mean_vector': mean_vector,
            'cov_matrix': cov_matrix
        }
        
        # Add the stats for this facies to the main dictionary
        stats_data['facies_stats'][facies_name] = current_facies_stats

    # --- 4. Calculate Global Plot Ranges ---
    print("\n\nCalculating global plot ranges...")
    for prop_name, all_values_list in plot_range_values.items():
        if len(all_values_list) == 0:
            print(f"  ERROR: No data for {prop_name} - cannot calculate range!")
            stats_data['plot_ranges'][prop_name] = {
                'min': np.nan,
                'max': np.nan
            }
            continue
            
        # Concatenate all facies data for this property into one big array
        all_values_flat = np.concatenate(all_values_list)
        
        # Remove any remaining NaN/Inf
        all_values_clean = all_values_flat[np.isfinite(all_values_flat)]
        
        if len(all_values_clean) == 0:
            print(f"  ERROR: No valid data for {prop_name} after cleaning!")
            global_min = np.nan
            global_max = np.nan
        else:
            # Find the min and max
            global_min = np.min(all_values_clean)
            global_max = np.max(all_values_clean)
        
        # Store the global min/max
        stats_data['plot_ranges'][prop_name] = {
            'min': global_min,
            'max': global_max
        }
        print(f"  {prop_name} range: {global_min:.2f} to {global_max:.2f} (n={len(all_values_clean)} valid points)")

    return stats_data

def save_statistics(stats_data, filename):
    """
    Saves the statistics dictionary to a file using pickle.
    
    Args:
        stats_data (dict): The dictionary returned by calculate_statistics.
        filename (str): The name of the file to save (e.g., 'facies_statistics.pkl').
    """
    with open(filename, 'wb') as f:
        pickle.dump(stats_data, f)
    print(f"\nSuccessfully saved statistics to: {filename}")

# --- Main execution ---
# This block runs when you execute 'statistics.py' directly
if __name__ == "__main__":
    
    # 1. Define the output file name
    output_filename = 'facies_statistics.pkl'
    
    print("=" * 60)
    print("Starting Statistics Calculation with Enhanced Debugging")
    print("=" * 60)
    
    # 2. Load all data from load_data.py
    print("\nStep 1: Loading data via load_data.py...")
    all_data = load_data()
    
    if 'facies' not in all_data or not all_data['facies']:
        print("\nError: No facies data found. Exiting.")
    else:
        # 3. Calculate all statistics
        print("\nStep 2: Calculating statistics...")
        statistics_data = calculate_statistics(all_data)
        
        # 4. Save the statistics to a .pkl file
        print("\nStep 3: Saving statistics...")
        save_statistics(statistics_data, output_filename)
        
        # 5. Print a summary to verify
        print("\n" + "=" * 60)
        print("Verification Summary")
        print("=" * 60)
        
        print("\nPlot Ranges:")
        for prop, ranges in statistics_data['plot_ranges'].items():
            print(f"  {prop}: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
        
        print("\nSample Facies Statistics (FaciesIIa):")
        try:
            example_stats = statistics_data['facies_stats']['FaciesIIa']
            for prop in ['Vp', 'Vs', 'Density', 'Ip', 'VpVs']:
                s = example_stats['univariate'][prop]
                print(f"  {prop}: mean={s['mean']:.4f}, std={s['std']:.4f}")
            print("\nVp/Vs Covariance Matrix:")
            print(example_stats['bivariate_VpVs']['cov_matrix'])
        except KeyError as e:
            print(f"Could not find data for verification: {e}")
        
        print("\n" + "=" * 60)
        print("Statistics Calculation Complete")
        print("=" * 60)
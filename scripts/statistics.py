"""
scripts/statistics.py

This module defines the statistical models for each facies.

FIXED VERSION: Now uses the exact same key names as the original.
"""

import numpy as np
import pickle
from scipy.stats import multivariate_normal

# ==============================================================================
#   Model 1: Rock Property Statistics (for Simulation)
# ==============================================================================

def compute_rock_property_statistics(facies_data_map, facies_names, bivariate_features, univariate_feature):
    """
    Calculates the statistical models (bivariate and univariate) for the
    rock properties needed for the simulation.
    
    IMPORTANT: This now uses the ORIGINAL key names:
    - 'mean_vector' (not 'mean')
    - 'cov_matrix' (not 'cov')

    Args:
        facies_data_map (dict): The dictionary of facies objects (raw_data['facies']).
        facies_names (list): An ordered list of facies names from config.py.
        bivariate_features (list): The two features for the bivariate model
                                   (e.g., ['Vp', 'Vs']).
        univariate_feature (str): The single feature for the univariate model
                                  (e.g., 'Density').

    Returns:
        dict: A dictionary where keys are facies names and values are another
              dictionary containing:
              'bivariate_VpVs': {'mean_vector': array(2), 'cov_matrix': array(2,2)}
              'univariate': {'Density': {'mean': float, 'std': float}}
    """
    
    stats_map = {}
    print("Calculating rock property statistics for simulation model...")
    print(f"  Bivariate features: {', '.join(bivariate_features)}")
    print(f"  Univariate feature: {univariate_feature}")

    for facies_name in facies_names:
        if facies_name not in facies_data_map:
            print(f"  Warning: Facies '{facies_name}' not found in loaded data. Skipping.")
            continue
            
        facies_obj = facies_data_map[facies_name]
        
        # Initialize structure to match original
        current_facies_stats = {
            'univariate': {},
            'bivariate_VpVs': {}
        }
        
        # --- 1. Compute Bivariate [Vp, Vs] Stats ---
        try:
            vp_data = getattr(facies_obj, bivariate_features[0])
            vs_data = getattr(facies_obj, bivariate_features[1])
            
            # Clean Vp and Vs independently first
            vp_clean = vp_data[np.isfinite(vp_data)]
            vs_clean = vs_data[np.isfinite(vs_data)]
            
            # Make sure they have the same length (matching original logic)
            min_len = min(len(vp_clean), len(vs_clean))
            vp_clean = vp_clean[:min_len]
            vs_clean = vs_clean[:min_len]
            
            if min_len > 0:
                # Create a 2xN array (matching original)
                vp_vs_stack = np.stack((vp_clean, vs_clean), axis=0)
                
                # Calculate covariance matrix
                cov_matrix = np.cov(vp_vs_stack)
                
                # Calculate means from the cleaned data
                vp_mean = np.mean(vp_clean)
                vs_mean = np.mean(vs_clean)
                mean_vector = np.array([vp_mean, vs_mean])
                
                # Also store univariate stats for Vp and Vs
                current_facies_stats['univariate']['Vp'] = {
                    'mean': vp_mean,
                    'std': np.std(vp_clean)
                }
                current_facies_stats['univariate']['Vs'] = {
                    'mean': vs_mean,
                    'std': np.std(vs_clean)
                }
            else:
                print(f"  Warning: Not enough valid [Vp, Vs] data for '{facies_name}'.")
                mean_vector = np.array([np.nan, np.nan])
                cov_matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
                current_facies_stats['univariate']['Vp'] = {'mean': np.nan, 'std': np.nan}
                current_facies_stats['univariate']['Vs'] = {'mean': np.nan, 'std': np.nan}

        except AttributeError as e:
            print(f"  Error getting bivariate data for {facies_name}: {e}")
            mean_vector = np.array([np.nan, np.nan])
            cov_matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            current_facies_stats['univariate']['Vp'] = {'mean': np.nan, 'std': np.nan}
            current_facies_stats['univariate']['Vs'] = {'mean': np.nan, 'std': np.nan}

        # --- 2. Compute Univariate [Density] Stats ---
        try:
            density_data = getattr(facies_obj, univariate_feature)
            clean_density_data = density_data[np.isfinite(density_data)]
            
            if clean_density_data.shape[0] > 0:
                density_mean = np.mean(clean_density_data)
                density_std = np.std(clean_density_data)
            else:
                print(f"  Warning: Not enough valid [Density] data for '{facies_name}'.")
                density_mean = np.nan
                density_std = np.nan

        except AttributeError as e:
            print(f"  Error getting univariate data for {facies_name}: {e}")
            density_mean = np.nan
            density_std = np.nan

        # Store Density stats
        current_facies_stats['univariate']['Density'] = {
            'mean': density_mean,
            'std': density_std
        }

        # --- 3. Store Bivariate Stats (with ORIGINAL key names) ---
        current_facies_stats['bivariate_VpVs'] = {
            'mean_vector': mean_vector,  # ← Original key name
            'cov_matrix': cov_matrix     # ← Original key name
        }
        
        # Store in main dict
        stats_map[facies_name] = current_facies_stats
        
        print(f"  Computed stats for: {facies_name} (n_bivariate={min_len if min_len > 0 else 0}, n_density={len(clean_density_data)})")
        
    print("Rock property statistics computation complete.")
    return stats_map


# ==============================================================================
#   Model 2: Classifier Feature Statistics (for Classification)
# ==============================================================================

def compute_classifier_statistics(ig_data_map, facies_names, classifier_features):
    """
    Calculates the final statistical model (mean, covariance, and inverse 
    covariance) for the classifier features ([Intercept, Gradient]) based 
    on the simulated data.

    Args:
        ig_data_map (dict): The dictionary of synthetic [I, G] pairs from
                            compute_intercept_gradient (e.g., ig_data['FaciesIIa']).
        facies_names (list): An ordered list of facies names from config.py.
        classifier_features (list): A list of feature names to use
                                    for the model (e.g., ['intercept', 'gradient']).

    Returns:
        dict: A dictionary where keys are facies names and values are another
              dictionary containing the 'mean' vector, 'cov' (covariance)
              matrix, and 'inv_cov' (inverse covariance) matrix for that facies.
    """
    
    # Validate input
    if ig_data_map is None:
        print("  ERROR: ig_data_map is None! Cannot compute classifier statistics.")
        print("  Make sure the intercept/gradient computation step completed successfully.")
        print("  Returning empty dict to allow pipeline to continue (will need to rerun).")
        return {}
    
    if not isinstance(ig_data_map, dict):
        print(f"  ERROR: ig_data_map is not a dictionary! Got type: {type(ig_data_map)}")
        print("  Returning empty dict to allow pipeline to continue.")
        return {}
    
    if len(ig_data_map) == 0:
        print("  ERROR: ig_data_map is empty! No intercept/gradient data found.")
        print("  Returning empty dict.")
        return {}
    
    stats_map = {}
    print("Calculating final classifier statistics from synthetic data...")
    print(f"  Features being used: {', '.join(classifier_features)}")
    print(f"  Processing {len(ig_data_map)} facies from I/G data")
    
    nan_matrix = np.full((len(classifier_features), len(classifier_features)), np.nan)

    for facies_name in facies_names:
        if facies_name not in ig_data_map:
            print(f"  Warning: Facies '{facies_name}' not found in simulated I/G data. Skipping.")
            continue
            
        facies_ig_data = ig_data_map[facies_name]
        
        # Stack the features into an (N, M) array, where M is number of features
        feature_data = []
        for feature in classifier_features:
            try:
                feature_data.append(facies_ig_data[feature])
            except KeyError:
                print(f"  Error: Feature '{feature}' not found in I/G data for '{facies_name}'.")
                feature_data.append(np.full_like(feature_data[0], np.nan) if feature_data else np.array([np.nan]))
        
        # Stack and transpose to get (N_points, N_features)
        data_stack = np.vstack(feature_data).T
        
        # Clean the stacked data: remove any rows containing NaN or Inf
        clean_data_stack = data_stack[np.all(np.isfinite(data_stack), axis=1)]
        
        if clean_data_stack.shape[0] < 2:
            print(f"  Warning: Not enough valid I/G data for '{facies_name}' (n={clean_data_stack.shape[0]}). Cannot compute covariance.")
            stats_map[facies_name] = {
                'mean': np.full(len(classifier_features), np.nan),
                'cov': nan_matrix,
                'inv_cov': nan_matrix
            }
            continue

        # Calculate the mean vector
        mean_vector = np.mean(clean_data_stack, axis=0)
        
        # Calculate the covariance matrix
        cov_matrix = np.cov(clean_data_stack, rowvar=False)
        
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print(f"  Error: Covariance matrix for {facies_name} is singular. Cannot invert.")
            inv_cov_matrix = nan_matrix
            
        stats_map[facies_name] = {
            'mean': mean_vector,
            'cov': cov_matrix,
            'inv_cov': inv_cov_matrix  # Store the new matrix
        }
        
        print(f"  Computed classifier stats for: {facies_name} (n={clean_data_stack.shape[0]})")
        
    print("Classifier statistics computation complete.")
    return stats_map


# ==============================================================================
#   Generic Helper Functions
# ==============================================================================

def save_statistics(stats_data, output_path):
    """
    Saves the statistics dictionary to a file using pickle.
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(stats_data, f)
        print(f"  Successfully saved statistics to: {output_path}")
    except Exception as e:
        print(f"  Error saving statistics to {output_path}: {e}")


def load_statistics(file_path):
    """
    Loads the cached statistics dictionary from a pickle file.
    """
    print(f"  Attempting to load cached statistics from: {file_path}")
    with open(file_path, 'rb') as f:
        stats_data = pickle.load(f)
    print(f"  Successfully loaded cached statistics.")
    return stats_data


def calculate_pdf(x, mean, cov):
    """
    Calculates the Probability Density Function (PDF) value for a data point.
    """
    try:
        # allow_singular=True is important as some data might be co-linear
        return multivariate_normal.pdf(x, mean=mean, cov=cov, allow_singular=True)
    except Exception as e:
        print(f"Error in PDF calculation: {e}. Mean: {mean}, Cov: {cov}")
        if hasattr(x, 'shape'):
            return np.zeros(x.shape[0])
        return 0
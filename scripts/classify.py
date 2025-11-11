"""
scripts/classify.py

This module contains the core classification mathematics.
It now supports multiple classification methods:
 1. Bayesian (original)
 2. Mahalanobis Distance

It is a "pure" worker module that performs calculations on
data passed to it. It does not read or write any files.

All plotting logic from the original classify.py is moved to visualization.py.
All file I/O is handled by main.py.
"""

import numpy as np

# We import the pre-existing calculate_pdf function from our
# statistics module.
from scripts.statistics import calculate_pdf

# ==============================================================================
#   Main Dispatch Function
# ==============================================================================

def run_classification(method, seismic_features, classifier_stats, facies_names, priors=None):
    """
    Acts as a router, calling the correct classification function based on
    the method specified in the config.
    
    Args:
        method (str): The name of the method to use (e.g., 'bayesian', 'mahalanobis').
        seismic_features (np.ndarray): The (N, 2) array of [Intercept, Gradient]
                                     data points to be classified.
        classifier_stats (dict): The dictionary of *final* classifier statistics
                               (mean/cov/inv_cov for [I, G] pairs).
        facies_names (list): The ordered list of facies names from config.py.
        priors (dict, optional): A dictionary mapping facies names to their
                                 prior probabilities. Only used by 'bayesian'.

    Returns:
        dict: A dictionary containing the classification results.
    """
    
    method_lower = method.lower()
    
    if method_lower == 'bayesian':
        print(f"Routing to Bayesian classifier...")
        if priors is None:
            raise ValueError("Bayesian classification method requires priors, but 'priors' argument was None.")
        return bayesian_classification(seismic_features, classifier_stats, facies_names, priors)
        
    elif method_lower == 'mahalanobis':
        print(f"Routing to Mahalanobis Distance classifier...")
        return mahalanobis_classification(seismic_features, classifier_stats, facies_names)
        
    else:
        raise ValueError(f"Unknown classification method: '{method}'. Supported methods are 'bayesian', 'mahalanobis'.")


# ==============================================================================
#   Method 1: Bayesian Classification
# ==============================================================================

def bayesian_classification(seismic_features, classifier_stats, facies_names, priors):
    """
    Performs Bayesian classification on the seismic feature array.
    
    P(Facies | Data) = [ P(Data | Facies) * P(Facies) ] / P(Data)
    
    Where:
     - P(Facies | Data) = Posterior probability (the result)
     - P(Data | Facies) = Likelihood (from calculate_pdf)
     - P(Facies)        = Prior probability
     - P(Data)          = Evidence (sum of all numerators)

    Args:
        (See run_classification for args)

    Returns:
        dict: A dictionary containing the classification results:
              - 'posterior_probs': (N, n_facies) array of all posterior probabilities.
              - 'most_likely_facies_idx': (N,) array of the *index* (0-8) of
                                          the most likely facies for each data point.
              - 'method': 'bayesian'
    """
    
    print("Starting Bayesian classification...")
    
    n_points = seismic_features.shape[0]
    n_facies = len(facies_names)
    
    # --- 1. Calculate Likelihoods and Numerators ---
    
    # Create empty arrays to hold our results
    # (N, n_facies) array for likelihood * prior
    numerators = np.zeros((n_points, n_facies))
    
    for i, facies_name in enumerate(facies_names):
        if facies_name not in classifier_stats:
            print(f"  Warning: No stats for {facies_name}. Skipping.")
            # Numerator remains 0 for this facies
            continue
            
        if facies_name not in priors:
            print(f"  Warning: No prior for {facies_name}. Skipping.")
            # Numerator remains 0 for this facies
            continue

        # Get the model for this facies
        mean = classifier_stats[facies_name]['mean']
        cov = classifier_stats[facies_name]['cov']
        
        # Get the prior for this facies
        prior = priors[facies_name]
        
        # Check for invalid stats before calculating
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
            print(f"  Warning: Invalid stats (NaN/Inf) for {facies_name}. Skipping.")
            numerators[:, i] = 0.0 # Ensure it's zero
            continue
            
        # P(Data | Facies)
        # Calculate likelihood for *all* seismic points at once
        likelihood = calculate_pdf(seismic_features, mean, cov)
        
        # P(Data | Facies) * P(Facies)
        # Store the numerator in the i-th column
        numerators[:, i] = likelihood * prior

    # --- 2. Calculate Denominator (Evidence) ---
    
    # P(Data) = Sum of all numerators for each data point
    # We sum along axis=1 (across all facies)
    # This gives a (N,) array of total probability for each data point
    denominator = np.sum(numerators, axis=1)
    
    # Avoid division by zero.
    # Where denominator is 0, set it to 1. The numerator will also be 0,
    # so the resulting probability will be 0, which is correct.
    denominator[denominator == 0] = 1.0

    # --- 3. Calculate Posterior Probabilities ---
    
    # P(Facies | Data) = Numerator / Denominator
    # We use (:, None) to broadcast the (N,) denominator array
    # for division against the (N, n_facies) numerator array.
    posterior_probs = numerators / denominator[:, None]
    
    # --- 4. Find Most Likely Facies ---
    
    # Get the *index* of the highest probability for each data point
    most_likely_facies_idx = np.argmax(posterior_probs, axis=1)
    
    print("Bayesian classification complete.")
    
    # Return all computed results
    results = {
        'posterior_probs': posterior_probs,
        'most_likely_facies_idx': most_likely_facies_idx,
        'method': 'bayesian'
    }
    
    return results


# ==============================================================================
#   Method 2: Mahalanobis Distance Classification
# ==============================================================================

def mahalanobis_classification(seismic_features, classifier_stats, facies_names):
    """
    Performs classification on the seismic feature array using Mahalanobis
    Distance. The class with the *minimum* distance is chosen as the winner.
    
    D^2 = (x - mu)^T * inv_cov * (x - mu)
    
    Where:
     - x         = The data point (or array of data points)
     - mu        = The mean vector for a given facies
     - inv_cov   = The inverse covariance matrix for that facies
     - D^2       = The Mahalanobis Distance (squared)
     
    Args:
        (See run_classification for args. 'priors' is not used.)

    Returns:
        dict: A dictionary containing the classification results:
              - 'mahalanobis_distances': (N, n_facies) array of all distances.
              - 'most_likely_facies_idx': (N,) array of the *index* (0-8) of
                                          the most likely facies for each data point.
              - 'method': 'mahalanobis'
    """
    
    print("Starting Mahalanobis Distance classification...")
    
    n_points = seismic_features.shape[0]
    n_facies = len(facies_names)
    
    # Store distances for each class in an (N_points, N_classes) array
    # We initialize with infinity, so any skipped class is automatically
    # a "loser" (i.e., has maximum distance).
    all_distances = np.full((n_points, n_facies), np.inf)

    for i, facies_name in enumerate(facies_names):
        if facies_name not in classifier_stats:
            print(f"  Warning: No stats for {facies_name}. Skipping.")
            continue
            
        stats = classifier_stats[facies_name]
        
        # Get the model for this facies
        mean_vec = stats['mean']
        inv_cov = stats['inv_cov']
        
        # Check for invalid stats before calculating
        if not (np.all(np.isfinite(mean_vec)) and np.all(np.isfinite(inv_cov))):
            print(f"  Warning: Invalid stats (NaN/Inf) for {facies_name}. Skipping.")
            continue
            
        # --- Vectorized Mahalanobis Distance Calculation ---
        
        # (x - mu)
        # diff is (N_points, N_features)
        diff = seismic_features - mean_vec
        
        # (x - mu) * inv_cov
        # temp is (N_points, N_features)
        temp = np.dot(diff, inv_cov)
        
        # (x - mu)^T * inv_cov * (x - mu)
        # We get D^2 for all pixels by summing the element-wise product
        # of diff and temp along the feature axis (axis=1).
        # d_squared is (N_points,)
        d_squared = np.sum(temp * diff, axis=1)
        
        all_distances[:, i] = d_squared
        
    # --- Find Most Likely Facies ---
    
    # Get the *index* of the *minimum* distance for each data point
    most_likely_facies_idx = np.argmin(all_distances, axis=1)
    
    print("Mahalanobis Distance classification complete.")
    
    # Return all computed results
    results = {
        'mahalanobis_distances': all_distances,
        'most_likely_facies_idx': most_likely_facies_idx,
        'method': 'mahalanobis'
    }
    
    return results
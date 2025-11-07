"""
scripts/classify.py

This module contains the core Bayesian classification mathematics.
It is a "pure" worker module that performs calculations on
data passed to it. It does not read or write any files.

All plotting logic from the original classify.py is moved to visualization.py.
All file I/O is handled by main.py.
"""

import numpy as np

# We import the pre-existing calculate_pdf function from our
# statistics module.
from scripts.statistics import calculate_pdf

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
        seismic_features (np.ndarray): The (N, 2) array of [Intercept, Gradient]
                                     data points to be classified.
        classifier_stats (dict): The dictionary of *final* classifier statistics
                               (mean/cov for [I, G] pairs).
        facies_names (list): The ordered list of facies names from config.py.
        priors (dict): A dictionary mapping facies names to their
                       prior probabilities (e.g., {'FaciesIIa': 0.11, ...}).

    Returns:
        dict: A dictionary containing the classification results:
              - 'posterior_probs': (N, n_facies) array of all posterior probabilities.
              - 'most_likely_facies_idx': (N,) array of the *index* (0-8) of
                                          the most likely facies for each data point.
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
        'most_likely_facies_idx': most_likely_facies_idx
    }
    
    return results
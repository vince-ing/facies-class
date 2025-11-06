"""
The classify.py module's job is to be the "classifier." It takes the observed data (the seismic I-G pairs) and the statistical models (the facies PDFs) and implements the Bayesian classification logic to determine the most likely facies for every single data point.


This module should be pure math. It shouldn't load files, and it shouldn't create plots. It takes arrays and dictionaries, performs calculations, and returns new arrays and dictionaries.

Here is the conceptual plan for refactoring scripts/classify.py:

Refactored: scripts/classify.py

Core Responsibility: To calculate the posterior probability P(Facies | Data) for every seismic data point and determine the most likely facies.

Principle: This module is the "brain." It implements Bayes' theorem by combining the statistical priors (from statistics.py) with the observed data (from feature_engineering.py). It returns the final classification results as data arrays.

Conceptual Functions:

1. run_classification(seismic_ig_data, facies_stats_map, facies_names)

    Purpose: The main "worker" function. It orchestrates the entire classification of all seismic data.

    Input:

        seismic_ig_data: The (N, 2) NumPy array of observed Intercept-Gradient pairs (from feature_engineering.load_ig_features()).

        facies_stats_map: The dictionary of statistical models (e.g., {'FaciesIIa': {'mean': ..., 'cov': ...}, ...}) (from statistics.load_statistics()).

        facies_names: The ordered list of names from config.py.

    Logic:

        Import the calculate_pdf function from statistics.py.

        Get n_points = len(seismic_ig_data) and n_facies = len(facies_names).

        Create an empty (n_points, n_facies) NumPy array called likelihood_array.

        Calculate Likelihoods P(Data | Facies):

            Loop i from 0 to n_facies:

                Get the facies_name, mean, and cov for the i-th facies.

                Calculate the PDF for all N points at once for this facies: likelihoods_for_this_facies = statistics.calculate_pdf(seismic_ig_data, mean, cov)

                Store this (n_points,) result in likelihood_array[:, i].

        Define Priors P(Facies):

            Assume equal priors for all facies: priors_array = numpy.full(n_facies, 1.0 / n_facies)

        Calculate Unscaled Posteriors P(Data | Facies) * P(Facies):

            posterior_unscaled = likelihood_array * priors_array (This uses NumPy broadcasting).

        Calculate Posteriors P(Facies | Data):

            Normalize each row (each data point) so its probabilities sum to 1: row_sums = posterior_unscaled.sum(axis=1, keepdims=True) posterior_probabilities = posterior_unscaled / row_sums

        Find Most Likely Facies:

            Find the index of the highest probability for each point: most_likely_indices = numpy.argmax(posterior_probabilities, axis=1)

    Output: A dictionary containing the full results:
    Python

    {
        "probability_map": posterior_probabilities,  # (n_points, n_facies) array
        "most_likely_map": most_likely_indices     # (n_points,) array
    }

2. group_facies(most_likely_indices, facies_names, grouping_scheme)

    Purpose: To create the "grouped" facies map (based on grouped_facies_map.png).

    Input:

        most_likely_indices: The (n_points,) array from run_classification.

        facies_names: The list from config.py.

        grouping_scheme: A dictionary (likely defined in config.py) that maps full facies names to grouped names (e.g., {'FaciesIIaOil': 'Oil', 'FaciesIIbOil': 'Oil', 'FaciesIIa': 'Brine', ...}).

    Logic:

        Uses numpy.vectorize or a fast mapping method to convert the array of indices (e.g., [0, 1, 2, 1]) into an array of grouped labels (e.g., ['Brine', 'Oil', 'Shale', 'Oil']).

    Output: A new (n_points,) array containing the grouped facies labels.

What this refactor achieves:

    Purity: This module is now a pure, non-parametric classifier. It's incredibly fast because all calculations are vectorized NumPy operations.

    Clear Separation: The logic is no longer mixed with plotting or data loading.

    Testability: You could easily write a test for run_classification by giving it a tiny seismic_ig_data array (e.g., 3 points) and a simple 2-facies facies_stats_map and checking the math.

    Flexible Output: It returns the full probability map, which is much more valuable than just the most likely map. visualization.py can use this to plot "confidence" or the probability of a specific facies.
"""
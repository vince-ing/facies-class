"""
Its core responsibility is to take the raw facies data and compute the statistical model for each facies. In this project, that model is the multivariate normal distribution (e.g., a 2D Gaussian) defined by a mean vector and a covariance matrix.

This statistical model is the "prior" that tells the classifier: "This is what FaciesIIa looks like, on average."

This module will be called by main.py after the data is loaded. Its output (facies_statistics.pkl) will be used by both the simulation.py (to draw samples) and classify.py (to calculate probabilities) modules.

Here is the conceptual plan for refactoring scripts/statistics.py:

Refactored: scripts/statistics.py

Core Responsibility: Compute, save, and load the statistical parameters (mean and covariance) for each facies. Provide the core probability calculation function (PDF).

Principle: This module owns the definition of the facies model. It computes the parameters and also provides the function to see how probable a data point is given those parameters.

Conceptual Functions:

1. compute_facies_statistics(facies_data_map)

    Purpose: The main "worker" function. To calculate the mean and covariance for every facies.

    Input: facies_data_map (the dictionary of arrays from load_data.py, e.g., {'FaciesIIa': array(...), ...}).

    Logic:

        Initializes an empty dictionary, stats_map.

        Loops through each facies_name, data_array in the facies_data_map.

        Calculates the mean vector using numpy.mean(data_array, axis=0).

        Calculates the covariance matrix using numpy.cov(data_array, rowvar=False).

        Stores these in the stats_map: stats_map[facies_name] = {'mean': mean, 'cov': cov}.

    Output: The completed stats_map dictionary.

2. save_statistics(stats_map, output_path)

    Purpose: To save the (potentially expensive) computed statistics to a pickle file for caching.

    Input:

        stats_map (the dictionary returned by compute_facies_statistics).

        output_path (e.g., config.FACIES_STATS_PATH).

    Logic:

        Uses pickle.dump() to save the stats_map object to the specified file path.

    Output: None (Side effect: creates a .pkl file).

3. load_statistics(file_path)

    Purpose: To load the cached statistics file. This allows main.py to skip the compute_facies_statistics step if the file already exists.

    Input: file_path (e.g., config.FACIES_STATS_PATH).

    Logic:

        Uses pickle.load() to open and read the file.

    Output: The stats_map dictionary.

4. calculate_pdf(x, mean, cov)

    Purpose: This is a crucial, reusable function. It calculates the Probability Density Function (PDF) value for a data point, given a statistical model.

    Input:

        x: A single data point (e.g., an [intercept, gradient] pair).

        mean: The mean vector for a specific facies.

        cov: The covariance matrix for that same facies.

    Logic:

        Uses scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=cov).

    Output: A single float (the probability density).

What this refactor achieves:

    Centralized Model: All the logic for defining the statistics (mean, cov) and using them (the PDF function) is in one place.

    Clear Caching: main.py can now use these functions in a clear "try to load, if fail, then compute" block.

    Modular: classify.py doesn't need to know how the PDF is calculated; it can just from scripts.statistics import calculate_pdf and use it.
"""
"""
we've agreed to rename from mc_draw.py to simulation.py.

This module's job is to create the prior model for the Bayesian classifier. It does this by taking the "blueprints" (the statistical models from statistics.py) and manufacturing a large set of labeled, simulated data points (the "Monte Carlo samples").

This simulated dataset (mc_samples.pkl) will be the "lookup table" that the classifier uses to answer the question: "How many of my simulated facies match this real seismic data point?"

Here is the conceptual plan for refactoring scripts/simulation.py:

Refactored: scripts/simulation.py

Core Responsibility: Generate a large, labeled set of data samples (Intercept, Gradient) for each facies by drawing from their respective statistical models (multivariate normal distributions).

Principle: This is the "sample factory." It takes the statistical models and a "sample count" and generates a large .pkl file of simulated data that will be used by the classifier.

Conceptual Functions:

1. run_monte_carlo(facies_stats_map, n_samples_per_facies, facies_names)

    Purpose: The main "worker" function. It generates a massive set of samples for all facies.

    Input:

        facies_stats_map: The dictionary from statistics.py (e.g., {'FaciesIIa': {'mean': ..., 'cov': ...}, ...}).

        n_samples_per_facies: The number of samples to draw per facies (e.g., config.MC_SAMPLE_COUNT).

        facies_names: The list of facies names from config.py, to ensure correct ordering.

    Logic:

        Create two empty lists: all_samples_list and all_labels_list.

        Loop through each facies_name in the facies_names list:

            Get the mean and cov from facies_stats_map[facies_name].

            Efficiently draw all samples at once: samples = numpy.random.multivariate_normal(mean, cov, size=n_samples_per_facies)

            This creates an (n_samples, 2) array. Append it to all_samples_list.

            Create a corresponding label array. We can use integer IDs for efficiency. labels = numpy.full(n_samples_per_facies, fill_value=facies_names.index(facies_name))

            Append this (n_samples,) array to all_labels_list.

        After the loop, stack all arrays into two giant arrays:

            final_samples = numpy.vstack(all_samples_list)

            final_labels = numpy.concatenate(all_labels_list)

    Output: A single, structured dictionary: {'samples': final_samples, 'labels': final_labels} (e.g., samples is (900000, 2) and labels is (900000,))

2. save_mc_samples(mc_data, output_path)

    Purpose: To save the (very large and expensive) simulation results to a pickle file for caching.

    Input:

        mc_data: The dictionary returned by run_monte_carlo.

        output_path (e.g., config.MC_SAMPLES_PATH).

    Logic:

        Uses pickle.dump() to save the mc_data dictionary.

    Output: None (Side effect: creates a .pkl file).

3. load_mc_samples(file_path)

    Purpose: To load the cached simulation file. This allows main.py to skip the run_monte_carlo step.

    Input: file_path (e.g., config.MC_SAMPLES_PATH).

    Logic:

        Uses pickle.load() to open and read the file.

    Output: The mc_data dictionary ({'samples': ..., 'labels': ...}).

What this refactor achieves:

    Efficiency: It uses numpy.random.multivariate_normal's size parameter to draw all samples at once, which is significantly faster than looping n_samples times.

    Structured Output: The output is a single, clean dictionary. The classify.py module gets exactly what it needs: a list of samples and a corresponding list of labels.

    Clarity: It's very clear that this module's one and only job is to run this simulation.
"""
"""
This module is the "processing" step. It's the first one that performs significant calculations. Its job is to take raw data (from load_data.py) and transform it into the features (the "axes" of our classification problem) that the subsequent modules will use.

In this project, it has two main jobs:

    Performing rock physics calculations (AVO) on the well data.

    Formatting the raw seismic data (Intercept, Gradient) into a structure the classifier can use.

This is the first module that will save its expensive-to-calculate results to the computed/ folder.

Here is the conceptual plan for refactoring scripts/feature_engineering.py:

Refactored: scripts/feature_engineering.py

Core Responsibility: Apply domain-specific (geophysics) calculations to transform raw data into "features" (AVO curves, I-G pairs) needed for modeling and classification.

Principle: This module translates raw data into meaningful features. It reads from data structures passed by main.py and writes new, processed .pkl files to the computed/ directory.

Conceptual Functions:

1. compute_and_save_avo(well_data, fluid_properties, output_path)

    Purpose: To perform the rock physics calculations, including fluid substitution and calculating AVO (Amplitude Versus Offset) reflectivity curves. This is likely based on the logic in your avopp.m resource file, using Zoeppritz or Aki-Richards approximations.

    Input:

        well_data: The DataFrame from load_data.load_well_data().

        fluid_properties: The dictionary from load_data.load_mineral_fluid_properties().

        output_path (e.g., config.AVO_DATA_PATH).

    Logic:

        Takes the well log Vp, Vs, and Density.

        Applies fluid substitution rules based on fluid_properties to model the different facies (e.g., "FaciesIIa" vs "FaciesIIaOil").

        Calculates the AVO reflectivity for a range of angles for each case.

        Saves the results (e.g., a dictionary of AVO curves) to output_path using pickle.dump().

    Output: The AVO data structure (which is also saved to a file).

2. compute_and_save_ig_features(seismic_data, output_path)

    Purpose: To format the raw 2D seismic maps into a 1D array of feature pairs that the classifier can use.

    Input:

        seismic_data: The dictionary of arrays from load_data.load_seismic_data().

        output_path (e.g., config.IG_DATA_PATH).

    Logic:

        Takes the 2D intercept array and 2D gradient array from the seismic_data input.

        Flattens each 2D array into a 1D array (e.g., intercept_flat = intercept_array.flatten()).

        Stacks them into a single (N, 2) array, where N is the total number of data points (pixels) in the map.

            ig_features = numpy.vstack((intercept_flat, gradient_flat)).T

        This (N, 2) array is the "observed data" that will be classified.

        Saves this array to output_path using pickle.dump().

    Output: The (N, 2) NumPy array of I-G features (which is also saved to a file).

3. load_avo_data(file_path)

    Purpose: To load the cached AVO data file.

    Input: file_path (e.g., config.AVO_DATA_PATH).

    Logic: Uses pickle.load().

    Output: The AVO data structure.

4. load_ig_features(file_path)

    Purpose: To load the cached Intercept-Gradient feature array.

    Input: file_path (e.g., config.IG_DATA_PATH).

    Logic: Uses pickle.load().

    Output: The (N, 2) NumPy array of I-G features.

What this refactor achieves:

    Isolation of Complexity: All the complex geophysics (AVO, fluid substitution) is now isolated in compute_and_save_avo.

    Data Shaping: The crucial step of transforming 2D maps into 1D feature vectors ((N, 2)) is now an explicit, clear function (compute_and_save_ig_features).

    Clear Caching: This module is the first one in the pipeline to create cached .pkl files in the computed/ folder, which main.py can use to skip steps on future runs.
"""
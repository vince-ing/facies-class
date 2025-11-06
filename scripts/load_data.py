"""
Core Responsibility: Read and parse all raw .txt files from the /data directory.

Principle: This module is "read-only" from the data/ folder. It takes file paths as arguments and returns data structures.

Conceptual Functions:

1. load_well_data(file_path)

    Purpose: To load the Well2.txt file.

    Input: file_path (e.g., config.WELL_2_PATH).

    Logic:

        Uses pandas.read_csv() with the correct separator (sep='\s+').

        Renames columns to be more programmatic (e.g., 'Vp/Vs' to VpVs).

        Performs any initial cleanup (like dropping empty rows, if any).

    Output: A clean Pandas DataFrame of the well log.

2. load_all_facies(directory_path, facies_names)

    Purpose: To load all 9 of the Facies*.txt files.

    Input:

        directory_path (e.g., config.FACIES_DIR_PATH).

        facies_names (e.g., config.FACIES_NAMES, a list like ['FaciesIIa', 'FaciesIIaOil', ...]).

    Logic:

        Loops through the facies_names list.

        For each name, constructs the full file path (e.g., .../data/FaciesTxtFiles/FaciesIIa.txt).

        Loads the file using numpy.loadtxt().

        Stores it in a dictionary.

    Output: A dictionary where keys are facies names and values are the loaded NumPy arrays.

        {'FaciesIIa': array(...), 'FaciesIIaOil': array(...), ...}

3. load_seismic_data(intercept_path, gradient_path, inline_path, xline_path)

    Purpose: To load the 4 seismic data files.

    Input: The four specific file paths from config.py.

    Logic:

        Loads each .txt file using numpy.loadtxt().

        Potentially reshapes them or packages them for convenience.

    Output: A dictionary or a custom object containing the four NumPy arrays (e.g., {'intercept': array(...), 'gradient': array(...), ...}).

4. load_mineral_fluid_properties(file_path)

    Purpose: To load the Mineral&FluidProperties.txt file.

    Input: file_path (e.g., config.MINERAL_FLUID_PATH).

    Logic:

        Parses this file (which might have a specific format) into a usable data structure.

    Output: A dictionary containing the properties.

What this refactor achieves:

    Single Responsibility: This file is now only responsible for loading raw data.

    Removed Functions: We would remove all functions that loaded .pkl files (like load_avo_data, load_facies_statistics, etc.). This logic belongs in main.py to control the pipeline flow (e.g., "try to load the computed file; if it fails, run the computation step").

    Testability: It's now very easy to test a function like load_well_data because it just takes a path and returns a DataFrame.
"""
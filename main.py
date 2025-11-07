"""
main.py - Main Pipeline Orchestrator

This is the single entry point for the entire facies classification pipeline.
It controls the "story" of the analysis by:
1.  Importing all settings from config.py.
2.  Importing "worker" functions from the scripts/ folder.
3.  Calling the worker functions in the correct order.
4.  Handling the caching of computed results (e.g., loading .pkl files
    if they exist, or running the computation if they don't).

To run the entire analysis, execute this file from your terminal:
    $ python main.py
"""

import os
import time
import pickle
import numpy as np

# Import all configuration settings from config.py
import config

# --- Import Worker Functions ---

# Step 1
from scripts.load_data import load_raw_data

# Step 2 & 5
from scripts.statistics import (
    compute_rock_property_statistics,
    compute_classifier_statistics,
    save_statistics, 
    load_statistics
)

# Step 3
from scripts.simulation import (
    run_avo_simulation, 
    save_avo_data, 
    load_avo_data
)

# Step 4
from scripts.feature_engineering import (
    compute_intercept_gradient,
    format_seismic_features,
    save_computed_data,
    load_computed_data
)

# Step 6
from scripts.classify import bayesian_classification

# Step 7
from scripts import visualization as viz

# --- Helper Function for Caching ---

def load_or_compute(file_path, compute_func, **compute_kwargs):
    """
    A generic helper to handle the caching logic.
    Tries to load a .pkl file. If it fails (missing or corrupt),
    it runs the provided compute_func, saves the result, and returns it.
    
    Args:
        file_path (str): The path to the .pkl file to load/save.
        compute_func (function): The function to run if loading fails.
        **compute_kwargs: Keyword arguments to pass to compute_func.
        
    Returns:
        The loaded or computed data.
    """
    try:
        # 1. Try to load
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"  Cached file not found: {file_path}")
            
        print(f"  Attempting to load cached data from: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Successfully loaded cached data.")
        
    except (FileNotFoundError, pickle.PickleError, EOFError) as e:
        # 2. If it fails, compute and save
        print(f"  {e}")
        print(f"  Running computation step...")
        
        # Run the computation function
        data = compute_func(**compute_kwargs)
        
        # Save the result
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"  Successfully computed and saved data to: {file_path}")
        except Exception as save_e:
            print(f"  Error saving computed data to {file_path}: {save_e}")
            
    return data

def clear_cache(computed_dir):
    """
    Deletes all .pkl files from the specified directory.
    """
    print(f"  Clearing cache from: {computed_dir}")
    if not os.path.exists(computed_dir):
        print(f"  Directory not found, nothing to clear.")
        return
    
    try:
        files_found = False
        for filename in os.listdir(computed_dir):
            if filename.endswith(".pkl"):
                files_found = True
                file_path = os.path.join(computed_dir, filename)
                os.remove(file_path)
                print(f"    Removed: {filename}")
        
        if not files_found:
            print(f"  No .pkl files found to remove.")
        else:
            print(f"  Cache cleared successfully.")
            
    except Exception as e:
        print(f"  Error clearing cache: {e}")

# --- Main Pipeline ---

def run_pipeline():
    """
    Executes the entire data processing and classification pipeline
    from start to finish.
    """
    start_time = time.time()
    print(f"Starting pipeline...")
    print(f"Loading configuration from config.py")
    
    # This dictionary will hold all the data as we process it
    pipeline_data = {}

    # ==========================================================================
    # STEP 1: LOAD RAW DATA
    # ==========================================================================
    print("\n--- STEP 1: Loading Raw Data ---")
    pipeline_data['raw_data'] = load_raw_data(
        facies_dir=config.FACIES_DIR,
        facies_column_names=config.FACIES_COLUMN_NAMES,
        seismic_paths=config.SEISMIC_FILE_PATHS,
        well_path=config.WELL_FILE_PATH,
        well_column_names=config.WELL_COLUMN_NAMES
    )
    print("--- STEP 1: Complete ---")

    # ==========================================================================
    # STEP 2: COMPUTE OR LOAD ROCK PROPERTY STATISTICS
    # ==========================================================================
    print("\n--- STEP 2: Loading/Computing Rock Property Statistics ---")
    
    # *** FIX: Passing the new, correct arguments ***
    pipeline_data['rock_property_stats'] = load_or_compute(
        file_path=config.ROCK_PROPERTY_STATS_PATH,
        compute_func=compute_rock_property_statistics,
        facies_data_map=pipeline_data['raw_data']['facies'],
        facies_names=config.FACIES_NAMES,
        bivariate_features=config.ROCK_PROPERTY_BIVARIATE_FEATURES,
        univariate_feature=config.ROCK_PROPERTY_UNIVARIATE_FEATURE
    )
    # *** END FIX ***
    
    print("--- STEP 2: Complete ---")

    # ==========================================================================
    # STEP 3: RUN MONTE CARLO & AVO SIMULATION
    # ==========================================================================
    print("\n--- STEP 3: Loading/Running AVO Simulation ---")

    # --- FIX: Update arguments to use dynamic top layer calculation ---
    pipeline_data['avo_data'] = load_or_compute(
        file_path=config.AVO_DATA_PATH,
        compute_func=run_avo_simulation,
        # Arguments for compute_func:
        facies_stats=pipeline_data['rock_property_stats'],
        facies_names=config.FACIES_NAMES,
        well_data=pipeline_data['raw_data']['well'], 
        n_samples=config.MC_SAMPLE_COUNT,
        theta_angles=config.AVO_THETA_ANGLES,
        top_layer_depth_start=config.TOP_LAYER_DEPTH_START,
        top_layer_depth_end=config.TOP_LAYER_DEPTH_END 
    )
    if pipeline_data['avo_data'] is None:
        print("  ERROR: AVO simulation returned None! Check Step 3.")
        exit(1)
    elif not isinstance(pipeline_data['avo_data'], dict):
        print(f"  ERROR: AVO data is not a dict! Type: {type(pipeline_data['avo_data'])}")
        exit(1)
    else:
        print(f"  AVO data contains {len(pipeline_data['avo_data'])} facies:")
        for facies_name, facies_avo in pipeline_data['avo_data'].items():
            print(f"    {facies_name}: {facies_avo.keys()}")
            if 'Rpp' in facies_avo:
                print(f"      Rpp shape: {facies_avo['Rpp'].shape}")

    print("--- STEP 3: Complete ---")

    # ==========================================================================
    # STEP 4: FEATURE ENGINEERING
    # ==========================================================================
    print("\n--- STEP 4a: Computing/Loading Synthetic [I, G] Pairs ---")
    
    # *** FIX: Using correct config variable IG_DATA_PATH ***
    pipeline_data['intercept_gradient_data'] = load_or_compute(
        file_path=config.IG_DATA_PATH,
        compute_func=compute_intercept_gradient,
        avo_data_map=pipeline_data['avo_data']
    )

    if pipeline_data['intercept_gradient_data'] is None:
        print("  ERROR: compute_intercept_gradient returned None!")
        print("  This means the AVO data structure is not correct.")
        exit(1)
    elif not isinstance(pipeline_data['intercept_gradient_data'], dict):
        print(f"  ERROR: I/G data is not a dict! Type: {type(pipeline_data['intercept_gradient_data'])}")
        exit(1)
    else:
        print(f"  I/G data contains {len(pipeline_data['intercept_gradient_data'])} facies:")
        for facies_name, facies_ig in pipeline_data['intercept_gradient_data'].items():
            print(f"    {facies_name}: {facies_ig.keys()}")
            if 'intercept' in facies_ig:
                print(f"      intercept length: {len(facies_ig['intercept'])}")
            if 'gradient' in facies_ig:
                print(f"      gradient length: {len(facies_ig['gradient'])}")
    # *** END FIX ***
    
    print("--- STEP 4a: Complete ---")
    
    print("\n--- STEP 4b: Computing/Loading Real Seismic Features ---")
    pipeline_data['seismic_features'] = load_or_compute(
        file_path=config.SEISMIC_FEATURES_PATH,
        compute_func=format_seismic_features,
        seismic_data_map=pipeline_data['raw_data']['seismic']
    )
    print("--- STEP 4b: Complete ---")

    # ==========================================================================
    # STEP 5: COMPUTE CLASSIFIER STATISTICS
    # ==========================================================================
    print("\n--- STEP 5: Loading/Computing Classifier Statistics ---")
    pipeline_data['classifier_stats'] = load_or_compute(
        file_path=config.CLASSIFIER_STATS_PATH,
        compute_func=compute_classifier_statistics,
        ig_data_map=pipeline_data['intercept_gradient_data'],
        facies_names=config.FACIES_NAMES,
        classifier_features=config.CLASSIFIER_FEATURES
    )
    print("--- STEP 5: Complete ---")

    # ==========================================================================
    # STEP 6: CLASSIFY SEISMIC DATA
    # ==========================================================================
    print("\n--- STEP 6: Loading/Running Bayesian Classification ---")
    
    # *** FIX: Using correct config variable FACIES_PRIORS ***
    pipeline_data['classification_results'] = load_or_compute(
        file_path=config.CLASSIFICATION_RESULTS_PATH,
        compute_func=bayesian_classification,
        seismic_features=pipeline_data['seismic_features'],
        classifier_stats=pipeline_data['classifier_stats'],
        facies_names=config.FACIES_NAMES,
        priors=config.FACIES_PRIORS
    )
    # *** END FIX ***
    
    print("--- STEP 6: Complete ---")

    # ==========================================================================
    # STEP 7: GENERATE PLOTS
    # ==========================================================================
    print("\n--- STEP 7: Generating All Plots ---")
    
    # *** FIX: Using correct config plot paths ***
    
    # a) Well Log Plot
    viz.plot_well_logs(
        well_data=pipeline_data['raw_data']['well'],
        output_path=config.WELL_LOG_PLOT_PATH
    )
    
    # b) PDF/CDF Plots
    viz.plot_property_distributions(
        facies_data_map=pipeline_data['raw_data']['facies'],
        facies_names=config.FACIES_NAMES,
        output_pdf_path=config.PDF_PLOT_PATH,
        output_cdf_path=config.CDF_PLOT_PATH
    )
    
    # c) AVO Reflectivity Curves
    viz.plot_reflectivity_curves(
        avo_data=pipeline_data['avo_data'],
        facies_names=config.FACIES_NAMES,
        output_path=config.AVO_CURVES_PLOT_PATH
    )
    
    # d) Intercept-Gradient Crossplots
    viz.plot_intercept_gradient(
        ig_data=pipeline_data['intercept_gradient_data'],
        seismic_features=pipeline_data['seismic_features'],
        facies_names=config.FACIES_NAMES,
        output_subplots=config.IG_SUBPLOTS_PATH,
        output_combined=config.IG_COMBINED_PLOT_PATH
    )
    
    # e) & f) Facies Classification Maps
    viz.plot_facies_maps(
        classification_results=pipeline_data['classification_results'],
        seismic_geometry=pipeline_data['raw_data']['seismic'],
        facies_names=config.FACIES_NAMES,
        output_most_likely=config.MOST_LIKELY_FACIES_MAP_PATH,
        output_grouped=config.GROUPED_FACIES_MAP_PATH
    )
    
    # *** END FIX ***
    
    print("--- STEP 7: Complete ---")

    # --- Pipeline Complete ---
    end_time = time.time()
    print(f"\n========================================================")
    print(f"  Pipeline finished in {end_time - start_time:.2f} seconds.")
    print(f"  All outputs are in '{config.COMPUTED_DIR}'")
    print(f"  All plots are in '{config.OUTPUT_DIR}'")
    print(f"========================================================")


if __name__ == "__main__":
    # This block ensures the pipeline runs only when the script is
    # executed directly.
    
    # Ensure the computed and output directories exist
    os.makedirs(config.COMPUTED_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Checks if cached data should be used or cleared 
    if not config.USE_CACHED_DATA:
        print(f"\n{'='*60}")
        print("WARNING: USE_CACHED_DATA is set to False.")
        print("Clearing all .pkl files from 'computed' directory...")
        clear_cache(config.COMPUTED_DIR)
        print(f"{'='*60}\n")
    else:
        print(f"\nINFO: USE_CACHED_DATA is set to True. Will use cached files.\n")
    
    # Normalize priors just in case they don't sum to 1
    try:
        total_prior = sum(config.FACIES_PRIORS.values())
        if not np.isclose(total_prior, 1.0):
            print(f"Warning: Priors sum to {total_prior:.2f}. Normalizing...")
            for name in config.FACIES_PRIORS:
                config.FACIES_PRIORS[name] /= total_prior
    except Exception as e:
        print(f"Error normalizing priors: {e}")

    
    run_pipeline()
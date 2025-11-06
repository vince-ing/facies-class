"""
Main entry point for the Facies Classification pipeline.

This script orchestrates the entire workflow from loading data to
classification and visualization.
"""

import os
import logging
import config  # Imports all your paths and parameters

# Import all the "worker" modules from the scripts package
# Note: You'll need to add an empty scripts/__init__.py file
from scripts import load_data
from scripts import feature_engineering  # (formerly compute_ig.py)
from scripts import statistics
from scripts import simulation           # (formerly mc_draw.py)
from scripts import classify
from scripts import visualization

# --- Setup Logging (Good Practice) ---
# (This is optional but highly recommended for a refactor)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_directories():
    """
    Ensures all output directories from the config file exist.
    """
    logger.info("Setting up output directories...")
    # Use os.makedirs with exist_ok=True to safely create directories
    # without errors if they already exist.
    os.makedirs(config.COMPUTED_DIR, exist_ok=True)
    os.makedirs(config.IMAGE_OUTPUT_DIR, exist_ok=True)
    logger.info("Directories are ready.")


def run_pipeline():
    """
    Orchestrates the full data processing and classification pipeline.
    """
    
    logger.info("Starting Facies Classification Pipeline...")

    # --- 1. SETUP ---
    setup_directories()

    # --- 2. LOAD DATA ---
    # Load all raw data from disk using functions from load_data.py
    logger.info("Loading raw data...")
    well_data = load_data.load_well_data(config.WELL_2_PATH)
    facies_data_map = load_data.load_all_facies(config.FACIES_DIR_PATH)
    seismic_data = load_data.load_seismic_data(
        intercept_path=config.SEISMIC_INTERCEPT_PATH,
        gradient_path=config.SEISMIC_GRADIENT_PATH
        # ... other seismic paths ...
    )
    logger.info("Raw data loaded successfully.")

    # --- 3. FEATURE ENGINEERING (Compute IG, AVO, etc.) ---
    # Process raw data into features needed for classification
    logger.info("Computing AVO and Intercept/Gradient data...")
    
    # This function would do the work and save the .pkl files
    feature_engineering.compute_and_save_avo(
        well_data,
        output_path=config.AVO_DATA_PATH
    )
    
    intercept_gradient_data = feature_engineering.compute_and_save_ig(
        seismic_data,
        output_path=config.IG_DATA_PATH
    )
    logger.info("Feature computation complete.")


    # --- 4. COMPUTE STATISTICS ---
    # Calculate statistical models (PDFs/CDFs) for each facies
    logger.info("Computing facies statistics...")
    facies_stats = statistics.compute_facies_statistics(facies_data_map)
    
    # Save the computed stats to a .pkl file
    statistics.save_statistics(facies_stats, config.FACIES_STATS_PATH)
    logger.info("Facies statistics saved.")


    # --- 5. RUN SIMULATION (Monte Carlo) ---
    # Generate samples based on the computed statistics
    logger.info("Running Monte Carlo simulation...")
    mc_samples = simulation.run_monte_carlo(
        facies_stats,
        n_samples=config.MC_SAMPLE_COUNT
    )
    
    # Save the samples to a .pkl file
    simulation.save_mc_samples(mc_samples, config.MC_SAMPLES_PATH)
    logger.info(f"Monte Carlo simulation complete with {config.MC_SAMPLE_COUNT} samples.")


    # --- 6. CLASSIFICATION ---
    # Run the Bayesian classification
    logger.info("Classifying seismic data...")
    classification_results = classify.run_classification(
        seismic_ig_data=intercept_gradient_data,
        mc_samples=mc_samples,
        facies_names=config.FACIES_NAMES
    )
    # classification_results might be a dictionary, e.g.:
    # { "most_likely_map": array, "probability_maps": dict }
    logger.info("Classification complete.")


    # --- 7. VISUALIZATION ---
    # Generate and save all plots and maps
    logger.info("Generating visualizations...")
    
    visualization.plot_well_log(
        well_data,
        output_path=os.path.join(config.IMAGE_OUTPUT_DIR, "well_log_plot.png")
    )
    
    visualization.plot_facies_pdfs(
        facies_stats,
        output_path=os.path.join(config.IMAGE_OUTPUT_DIR, "facies_pdf_plots.png")
    )
    
    visualization.plot_most_likely_map(
        classification_results["most_likely_map"],
        output_path=os.path.join(config.IMAGE_OUTPUT_DIR, "most_likely_facies_map.png")
    )
    
    # ... add other plotting functions ...
    
    logger.info("Visualizations saved to image output folder.")
    logger.info("--- PIPELINE FINISHED SUCCESSFULLY ---")


# --- Standard Python entry point ---
if __name__ == "__main__":
    # This block ensures the code only runs when you execute
    # "python main.py" from your terminal.
    run_pipeline()
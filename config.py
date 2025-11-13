"""
config.py - Project Configuration File

This file serves as the single "Control Panel" for the entire facies classification pipeline.
It defines all user-configurable parameters, file paths, and settings that are 
used by the orchestrator (main.py) and passed to the worker scripts.

To change a parameter (e.g., number of Monte Carlo samples, file paths, plot names), 
you should only need to edit this file.
"""

import os
import numpy as np

# ==============================================================================
#   DIRECTORY & FILE PATHS
# ==============================================================================
# --- Base Directories ---
# Use os.path.dirname(__file__) to get the directory of this config file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
COMPUTED_DIR = os.path.join(BASE_DIR, 'computed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'image output')

# --- Raw Data Paths ---
FACIES_DIR = os.path.join(DATA_DIR, 'FaciesTxtFiles')
SEISMIC_DIR = os.path.join(DATA_DIR, 'Seismic')
WELL_FILE_PATH = os.path.join(DATA_DIR, 'Well2.txt')

# Dictionary mapping keys to their full file paths
SEISMIC_FILE_PATHS = {
    'inline': os.path.join(SEISMIC_DIR, 'SeismicInlineNumbers.txt'),
    'xline': os.path.join(SEISMIC_DIR, 'SeismicXlineNumers.txt'),
    'intercept': os.path.join(SEISMIC_DIR, 'SeismicIntercept.txt'),
    'gradient': os.path.join(SEISMIC_DIR, 'SeismicGradient.txt')
}

# --- Computed (Cached) File Paths ---
# These paths are for the .pkl files created by the pipeline
ROCK_PROPERTY_STATS_PATH = os.path.join(COMPUTED_DIR, 'rock_property_statistics.pkl')
AVO_DATA_PATH = os.path.join(COMPUTED_DIR, 'avo_data.pkl')
IG_DATA_PATH = os.path.join(COMPUTED_DIR, 'intercept_gradient_data.pkl')
SEISMIC_FEATURES_PATH = os.path.join(COMPUTED_DIR, 'seismic_features.pkl')
CLASSIFIER_STATS_PATH = os.path.join(COMPUTED_DIR, 'classifier_statistics.pkl')
CLASSIFIER_STATS_PATH = os.path.join(COMPUTED_DIR, 'classifier_statistics.pkl')

# --- Output Plot File Paths ---
WELL_LOG_PLOT_PATH = os.path.join(OUTPUT_DIR, 'well_log_plot.png')
PDF_PLOT_PATH = os.path.join(OUTPUT_DIR, 'facies_pdf_plots.png')
CDF_PLOT_PATH = os.path.join(OUTPUT_DIR, 'facies_cdf_plots.png')
AVO_CURVES_PLOT_PATH = os.path.join(OUTPUT_DIR, 'avo_reflectivity_curves.png')
IG_SUBPLOTS_PATH = os.path.join(OUTPUT_DIR, 'intercept_gradient_subplots.png')
IG_COMBINED_PLOT_PATH = os.path.join(OUTPUT_DIR, 'intercept_gradient_combined.png')
MOST_LIKELY_FACIES_MAP_PATH = os.path.join(OUTPUT_DIR, 'most_likely_facies_map.png')
GROUPED_FACIES_MAP_PATH = os.path.join(OUTPUT_DIR, 'grouped_facies_map.png')


# ==============================================================================
#   DATA LOADING PARAMETERS
# ==============================================================================

# Column names for the 9 facies .txt files
FACIES_COLUMN_NAMES = [
    'Depth', 'Vp', 'Vs', 'Density', 'GR', 'Porosity',
    'Ip', 'Is', 'VpVs', 'Vclay', 'Sw', 'Sxo'
]

# Column names for the Well2.txt file
WELL_COLUMN_NAMES = [
    'Depth', 'Vp', 'Vs', 'Density', 'GR', 'Porosity', 'Vclay'
]

# ==============================================================================
#   PIPELINE PARAMETERS
# ==============================================================================

# --- Cache Behavior ---
# Set to False to clear all .pkl files from the 'computed' directory
# before running the pipeline, forcing a full re-computation.
USE_CACHED_DATA = False

# --- Facies / Lithology ---
# The names of the facies, in the order they will be processed.
# This is crucial for ensuring statistics, simulations, and plots
# all use the same order and labels.
FACIES_NAMES = [
    'FaciesIIaOil', 'FaciesIIbOil', 'FaciesIIcOil',
    'FaciesIIa',     'FaciesIIb',     'FaciesIIc',
    'FaciesIII',     'FaciesIV',      'FaciesV'
]

# --- Step 2: Rock Property Statistics ---
# **THIS SECTION IS UPDATED TO FIX THE BUG**
# We are now defining the Bivariate and Univariate features separately,
# as required by the corrected scripts/statistics.py module.
ROCK_PROPERTY_BIVARIATE_FEATURES = ['Vp', 'Vs']
ROCK_PROPERTY_UNIVARIATE_FEATURE = 'Density'

# --- Step 3: AVO Simulation ---
MC_SAMPLE_COUNT = 5000

# Angle range for reflectivity calculation 
AVO_THETA_ANGLES = np.arange(0, 30, 1)

# Constant properties for the "Top Layer" (Layer 4)
# Used in the avopp (Aki-Richards) simulation
TOP_LAYER_DEPTH_START = 2143.2  # m
TOP_LAYER_DEPTH_END = 2154.1    # m

# --- Step 4 & 5: Classifier Features ---
# The features that will be used for the final classification
# (These must match the keys in scripts/feature_engineering.py)
CLASSIFIER_FEATURES = ['intercept', 'gradient']

# --- Step 6: Bayesian Classification ---

# --- Classification Method ---
# Select the algorithm to use for classification.
# Options: 'bayesian', 'mahalanobis'
CLASSIFICATION_METHOD = 'mahalanobis'

# --- Classification Results Path (Dynamic) ---
# This path is now built dynamically based on the method chosen above.
# e.g., 'classification_results_bayesian.pkl'
# or 'classification_results_mahalanobis.pkl'
CLASSIFICATION_RESULTS_PATH = os.path.join(
    COMPUTED_DIR, 
    f'classification_results_{CLASSIFICATION_METHOD.lower()}.pkl'
)


# Prior probabilities for each facies (must sum to 1.0)
FACIES_PRIORS = {
    'FaciesIIa': 0.11,
    'FaciesIIaOil': 0.11,
    'FaciesIIb': 0.11,
    'FaciesIIbOil': 0.11,
    'FaciesIIc': 0.11,
    'FaciesIIcOil': 0.11,
    'FaciesIII': 0.11,
    'FaciesIV': 0.11,
    'FaciesV': 0.11  # Note: This is 0.11 * 9 = 0.99. The script will normalize.
}

FACIES_COLORS = {
    'FaciesIIa': '#8B0000',     # dark red
    'FaciesIIaOil': '#FF0000',  # red
    'FaciesIIb': '#FFA500',     # orange
    'FaciesIIbOil': '#FFD700',  # gold
    'FaciesIIc': '#FF00FF',     # magenta
    'FaciesIIcOil': '#EE82EE',  # violet
    'FaciesIII': '#008000',    # green
    'FaciesIV': '#00FFFF',     # cyan
    'FaciesV': '#0000FF',      # blue
}


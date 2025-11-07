"""
scripts/load_data.py

This module is responsible for loading the raw .txt data files from disk.
It is a "worker" module intended to be imported and called by main.py.
It does not contain any hard-coded paths or parameters; all configuration
is passed in as arguments.
"""

import os
import glob
import numpy as np
import pandas as pd

class Facies:
    """
    A simple class to act as a 'struct' for holding facies data.
    This allows accessing data using dot notation (e.g., facies.Vp).
    """
    pass

def load_raw_data(facies_dir, facies_column_names, seismic_paths, well_path, well_column_names):
    """
    Loads all raw project data from .txt files.
    
    This function is designed to be called by main.py, which supplies all
    necessary file paths and configuration parameters.

    Args:
        facies_dir (str): Path to the 'FaciesTxtFiles' directory.
        facies_column_names (list): List of column names for facies files.
        seismic_paths (dict): A dictionary mapping seismic data keys 
                              (e.g., 'intercept') to their full file paths.
        well_path (str): Path to the 'Well2.txt' file.
        well_column_names (list): List of column names for the well file.

    Returns:
        dict: A nested dictionary containing all project data.
    """
    
    # This is the main dictionary that will hold all our data
    data = {}
    
    # ======================================================================
    # 1. Load Facies Data
    # ======================================================================
    print("Loading facies data...")
    data['facies'] = {}
    
    # Get a list of all facies .txt files from the directory
    facies_files = glob.glob(os.path.join(facies_dir, '*.txt'))
    
    if not facies_files:
        print(f"Warning: No .txt files found in '{facies_dir}'. Please check the path.")
        
    for f_path in facies_files:
        # Extract the facies name from the filename (e.g., "FaciesIIa")
        facies_name = os.path.basename(f_path).replace('.txt', '')
        
        try:
            # Load the data using pandas.
            df = pd.read_csv(
                f_path,
                sep='\t',
                comment='%',  # This correctly skips the header line that starts with %
                header=None,
                names=facies_column_names
            )
            
            # Create a new Facies object
            facies_obj = Facies()
            
            # Populate the object with the data as numpy arrays
            for col_name in facies_column_names:
                setattr(facies_obj, col_name, df[col_name].values)
            
            # Store the object in our main data dictionary
            data['facies'][facies_name] = facies_obj
            print(f"  Successfully loaded: {facies_name} ({len(df)} data points)")

        except Exception as e:
            print(f"  Error loading {f_path}: {e}")
            
    print(f"\nTotal facies loaded: {len(data['facies'])}\n")

    # ======================================================================
    # 2. Load Seismic Data
    # ======================================================================
    print("Loading seismic data...")
    data['seismic'] = {}
    
    for key, f_path in seismic_paths.items():
        try:
            data['seismic'][key] = np.loadtxt(f_path)
            print(f"  Successfully loaded: '{f_path}' (shape: {data['seismic'][key].shape})")
        except FileNotFoundError:
            print(f"  File not found: '{f_path}'. Skipping.")
        except Exception as e:
            print(f"  Error loading {f_path}: {e}")
    print("\n")

    # ======================================================================
    # 3. Load Well2.txt Data
    # ======================================================================
    print("Loading Well2.txt data...")
    data['well'] = {}
    
    try:
        # Load the main well file
        well_df = pd.read_csv(
            well_path, 
            sep=r'\s+', 
            comment='%', 
            header=None,
            names=well_column_names
        )
        
        # Store data in the dictionary as numpy arrays
        for col in well_df.columns:
            data['well'][col] = well_df[col].values
            
        print(f"  Successfully loaded: {well_path}")
        print("\n--- Well2.txt Head (for inspection) ---")
        print(well_df.head())
        print("\n")
        
    except FileNotFoundError:
        print(f"  File not found: '{well_path}'. Skipping.\n")
    except Exception as e:
        print(f"  Error loading {well_path}: {e}\n")

    print("="*30)
    print("Data loading complete.")
    print(f"Total facies in data: {list(data['facies'].keys())}")
    print(f"Seismic data keys: {list(data['seismic'].keys())}")
    print(f"Well2 data keys: {list(data['well'].keys())}")
    print("="*30)
    
    return data
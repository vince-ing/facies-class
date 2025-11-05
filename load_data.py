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

def load_data(base_dir='.'):
    """
    Loads all project data from the .txt files using the specified
    directory structure.
    
    This function loads:
    1.  All 9 facies files from the 'FaciesTxtFiles' directory.
    2.  All 4 seismic files from the 'Seismic' directory.
    3.  The main 'Well2.txt' log file from the base directory.
    
    Args:
        base_dir (str): The root directory of the project. Defaults to '.'.

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
    
    # Define the clean column names for the facies files
    facies_column_names = [
        'Depth', 'Vp', 'Vs', 'Density', 'GR', 'Porosity', 
        'Ip', 'Is', 'VpVs', 'Vclay', 'Sw', 'Sxo'
    ]
    
    # Get a list of all facies .txt files from the 'FaciesTxtFiles' subdirectory
    facies_dir = os.path.join(base_dir, 'FaciesTxtFiles')
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
    seismic_dir = os.path.join(base_dir, 'Seismic')
    
    # Define file paths
    seismic_files = {
        'inline': os.path.join(seismic_dir, 'SeismicInlineNumbers.txt'),
        'xline': os.path.join(seismic_dir, 'SeismicXlineNumers.txt'),
        'intercept': os.path.join(seismic_dir, 'SeismicIntercept.txt'),
        'gradient': os.path.join(seismic_dir, 'SeismicGradient.txt')
    }

    for key, f_path in seismic_files.items():
        try:
            data['seismic'][key] = np.loadtxt(f_path)
            print(f"  Successfully loaded: '{f_path}' (shape: {data['seismic'][key].shape})")
        except FileNotFoundError:
            print(f"  File not found: '{f_path}'. Skipping.")
        except Exception as e:
            print(f"  Error loading {f_path}: {e}")
    print("\n")

    # ======================================================================
    # 3. Load Well2.txt Data (*** THIS SECTION IS UPDATED ***)
    # ======================================================================
    print("Loading Well2.txt data...")
    data['well'] = {}
    well_file_path = os.path.join(base_dir, 'Well2.txt')
    
    # Define the 7 column names from the header you provided
    well_column_names = ['Depth', 'Vp', 'Vs', 'Density', 'GR', 'Porosity', 'Vclay']
    
    try:
        # Load the main well file from the base directory
        # We now provide the correct parameters:
        #   sep='\s+'          - Replaces delim_whitespace and handles any whitespace
        #   comment='%'        - Skips the header line
        #   header=None        - Tells pandas there is no header row to read
        #   names=...          - Provides the correct column names
        well_df = pd.read_csv(
            well_file_path, 
            sep=r'\s+', 
            comment='%', 
            header=None,
            names=well_column_names
        )
        
        # Store data in the dictionary as numpy arrays
        for col in well_df.columns:
            data['well'][col] = well_df[col].values
            
        print(f"  Successfully loaded: {well_file_path}")
        print("\n--- Well2.txt Head (for inspection) ---")
        print(well_df.head())
        print("\n")
        
    except FileNotFoundError:
        print(f"  File not found: '{well_file_path}'. Skipping.\n")
    except Exception as e:
        print(f"  Error loading {well_file_path}: {e}\n")

    print("="*30)
    print("Data loading complete.")
    print(f"Total facies in data: {list(data['facies'].keys())}")
    print(f"Seismic data keys: {list(data['seismic'].keys())}")
    print(f"Well2 data keys: {list(data['well'].keys())}") # This output should be fixed
    print("="*30)
    
    return data


# --- Main execution ---
# When you run this, 'all_data' will hold all loaded data.
if __name__ == "__main__":
    
    all_data = load_data()

    # Example of how you can access the data (will only work if loading was successful)
    try:
        if all_data['facies']:
            # Get the first available facies to show as an example
            first_facies_name = list(all_data['facies'].keys())[0]
            print(f"\nExample data access for '{first_facies_name}':")
            print(f"  Vp (first 5 values): {all_data['facies'][first_facies_name].Vp[:5]}")
            print(f"  Density (first 5 values): {all_data['facies'][first_facies_name].Density[:5]}")
        else:
            print("\nNo facies data was loaded, cannot show example.")

        if 'intercept' in all_data['seismic']:
            print(f"\nExample data access for 'seismic':")
            print(f"  Intercept (shape): {all_data['seismic']['intercept'].shape}")
        else:
            print("\nNo seismic data was loaded, cannot show example.")

    except Exception as e:
        print(f"\nError showing data examples: {e}")
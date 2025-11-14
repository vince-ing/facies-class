"""
make_animation.py

A standalone script to generate animated GIFs by iterating over
a specified parameter in the facies classification pipeline.

This script imports the worker functions from the 'scripts' folder
and mimics the 'main.py' pipeline, but is designed to be run
iteratively to generate frames for an animation.

To use:
1.  Configure the animation settings (PARAM_TO_ANIMATE, MIN, MAX, STEP)
    in the `if __name__ == "__main__":` block.
2.  Run this file directly: $ python make_animation.py
"""

import os
import shutil
import pickle
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont # <-- MODIFIED: Added ImageDraw, ImageFont

# --- Import all configuration settings from config.py ---
# We will MODIFY these settings in memory, not just read them
import config

# --- Import Worker Functions (same as main.py) ---
from scripts.load_data import load_raw_data
from scripts.statistics import (
    compute_rock_property_statistics,
    compute_classifier_statistics
)
from scripts.simulation import run_avo_simulation
from scripts.feature_engineering import (
    compute_intercept_gradient,
    format_seismic_features
)
from scripts.classify import run_classification
from scripts import visualization as viz

# --- Caching Helper (copied from main.py) ---

def load_or_compute(file_path, compute_func, **compute_kwargs):
    """
    A generic helper to handle the caching logic.
    Tries to load a .pkl file. If it fails (missing or corrupt),
    it runs the provided compute_func, saves the result, and returns it.
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

# --- Animation-specific cache invalidation ---

def invalidate_cache(param_name):
    """
    Deletes cached .pkl files that are downstream of a changed parameter.
    This forces load_or_compute to re-run those steps.
    """
    print(f"  Invalidating cache due to change in: {param_name}")
    
    # These files depend on the AVO simulation (MC_SAMPLE_COUNT, AVO_THETA_ANGLES)
    avo_dependent_files = [
        config.AVO_DATA_PATH,
        config.IG_DATA_PATH,
        config.CLASSIFIER_STATS_PATH,
        config.CLASSIFICATION_RESULTS_PATH
    ]
    
    # This file only depends on the classification method
    class_dependent_files = [
        config.CLASSIFICATION_RESULTS_PATH
    ]

    files_to_delete = []
    if param_name in ['MC_SAMPLE_COUNT', 'AVO_THETA_ANGLES']:
        files_to_delete = avo_dependent_files
    elif param_name == 'CLASSIFICATION_METHOD':
        files_to_delete = class_dependent_files
    
    for f_path in files_to_delete:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                print(f"    Removed cached file: {f_path}")
            except Exception as e:
                print(f"    Error removing {f_path}: {e}")
        else:
            # This is fine, just means it would have re-computed anyway
            print(f"    Cached file not found, no need to remove: {f_path}")


# --- NEW: Helper function to format overlay text ---
def _get_text_for_frame(param_name, value):
    """Formats the text overlay for a given parameter and value."""
    if param_name == 'MC_SAMPLE_COUNT':
        return f"MC Sample Count: {value}"
    elif param_name == 'AVO_THETA_ANGLES':
        # --- FIX for zero-size array ---
        if value.size == 0:
            return "Max Angle: N/A (empty array)"
        # --- END FIX ---
        elif hasattr(value, 'max'):
            # The max value in np.arange(0, 30, 1) is 29.
            # This text will correctly show "Max Angle: 29 degrees"
            return f"Max Angle: {value.max()} degrees"
        else:
            return f"Max Angle: {value} degrees" # Fallback
    elif param_name == 'CLASSIFICATION_METHOD':
        return f"Method: {str(value).title()}"
    return f"{param_name}: {value}"


# --- Main Animation Function ---

def create_animation(param_to_animate, values, frame_duration_ms=500):
    """
    Generates an animation by iterating a parameter over a list of values.
    
    Args:
        param_to_animate (str): The name of the config variable to change
                                (e.g., 'MC_SAMPLE_COUNT').
        values (list): A list of values to iterate over for that parameter.
        frame_duration_ms (int): Duration of each frame in the GIF.
    """
    
    start_time = time.time()
    print(f"--- Starting Animation ---")
    print(f"  Animating Param: {param_to_animate}")
    print(f"  Frames to generate: {len(values)}")
    if len(values) > 5:
        print(f"  Values (first 5): {values[:5]}...")
    else:
        print(f"  Values: {values}")
    print(f"--------------------------")

    # --- Setup output directories ---
    temp_dir = os.path.join(config.BASE_DIR, "gif_output", "temp_frames")
    final_dir = os.path.join(config.BASE_DIR, "gif_output")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    frame_paths = []
    pipeline_data = {}
    
    # --- Load "static" data (Steps 1, 2, 4b) ---
    # These steps are independent of the animated parameters
    # and will use the cache (or generate it once).
    print("\n--- Loading static data (Steps 1, 2, 4b) ---")
    try:
        # Step 1: Load Raw Data (no caching, same as main.py)
        print("  Running Step 1: Load Raw Data...")
        pipeline_data['raw_data'] = load_raw_data(
            facies_dir=config.FACIES_DIR,
            facies_column_names=config.FACIES_COLUMN_NAMES,
            seismic_paths=config.SEISMIC_FILE_PATHS,
            well_path=config.WELL_FILE_PATH,
            well_column_names=config.WELL_COLUMN_NAMES
        )
        
        # Step 2: Load/Compute Rock Stats (uses cache)
        print("  Running Step 2: Load/Compute Rock Stats...")
        pipeline_data['rock_property_stats'] = load_or_compute(
            file_path=config.ROCK_PROPERTY_STATS_PATH,
            compute_func=compute_rock_property_statistics,
            facies_data_map=pipeline_data['raw_data']['facies'],
            facies_names=config.FACIES_NAMES,
            bivariate_features=config.ROCK_PROPERTY_BIVARIATE_FEATURES,
            univariate_feature=config.ROCK_PROPERTY_UNIVARIATE_FEATURE
        )
        
        # Step 4b: Load/Compute Seismic Features (uses cache)
        print("  Running Step 4b: Load/Compute Seismic Features...")
        pipeline_data['seismic_features'] = load_or_compute(
            file_path=config.SEISMIC_FEATURES_PATH,
            compute_func=format_seismic_features,
            seismic_data_map=pipeline_data['raw_data']['seismic']
        )
        print("--- Static data loaded successfully. ---")
    except Exception as e:
        print(f"FATAL ERROR: Could not load static data. {e}")
        print("Please run main.py once to generate the cached files for Steps 2 & 4b.")
        return

    # --- Main Frame Generation Loop ---
    for i, value in enumerate(values):
        print(f"\n========================================================")
        print(f"  Generating Frame {i+1} / {len(values)}")
        
        # --- FIX for zero-size array ---
        # Special handling for AVO_THETA_ANGLES to show max angle
        if param_to_animate == 'AVO_THETA_ANGLES':
            if value.size == 0:
                print(f"  Setting {param_to_animate} = [empty array] (Max Angle: N/A)")
            else:
                print(f"  Setting {param_to_animate} = 0 to {value.max()} degrees")
        else:
            print(f"  Setting {param_to_animate} = {value}")
        # --- END FIX ---
            
        print(f"========================================================")
        
        # 1. Modify the config *in memory*
        if param_to_animate == 'MC_SAMPLE_COUNT':
            config.MC_SAMPLE_COUNT = value
        elif param_to_animate == 'AVO_THETA_ANGLES':
            config.AVO_THETA_ANGLES = value
        elif param_to_animate == 'CLASSIFICATION_METHOD':
            config.CLASSIFICATION_METHOD = value
        
        # 2. Invalidate cache for downstream steps
        invalidate_cache(param_to_animate)
        
        # 3. Run the rest of the pipeline
        try:
            # --- STEP 3: AVO SIMULATION ---
            print("\n--- Running Step 3: AVO Simulation ---")
            # Note: This step might fail if value.size == 0.
            # If it does, we will catch the error and skip the frame.
            pipeline_data['avo_data'] = load_or_compute(
                file_path=config.AVO_DATA_PATH,
                compute_func=run_avo_simulation,
                facies_stats=pipeline_data['rock_property_stats'],
                facies_names=config.FACIES_NAMES,
                well_data=pipeline_data['raw_data']['well'], 
                n_samples=config.MC_SAMPLE_COUNT, # Uses modified config
                theta_angles=config.AVO_THETA_ANGLES, # Uses modified config
                top_layer_depth_start=config.TOP_LAYER_DEPTH_START,
                top_layer_depth_end=config.TOP_LAYER_DEPTH_END 
            )

            # --- STEP 4a: [I, G] Pairs ---
            print("\n--- Running Step 4a: [I, G] Pairs ---")
            pipeline_data['intercept_gradient_data'] = load_or_compute(
                file_path=config.IG_DATA_PATH,
                compute_func=compute_intercept_gradient,
                avo_data_map=pipeline_data['avo_data']
            )
            
            # --- STEP 5: Classifier Stats ---
            print("\n--- Running Step 5: Classifier Stats ---")
            pipeline_data['classifier_stats'] = load_or_compute(
                file_path=config.CLASSIFIER_STATS_PATH,
                compute_func=compute_classifier_statistics,
                ig_data_map=pipeline_data['intercept_gradient_data'],
                facies_names=config.FACIES_NAMES,
                classifier_features=config.CLASSIFIER_FEATURES
            )

            # --- STEP 6: Classification ---
            print("\n--- Running Step 6: Classification ---")
            pipeline_data['classification_results'] = load_or_compute(
                file_path=config.CLASSIFICATION_RESULTS_PATH,
                compute_func=run_classification,
                method=config.CLASSIFICATION_METHOD, # Uses modified config
                seismic_features=pipeline_data['seismic_features'],
                classifier_stats=pipeline_data['classifier_stats'],
                facies_names=config.FACIES_NAMES,
                priors=config.FACIES_PRIORS
            )
            
            # --- STEP 7: Generate Plot ---
            print("\n--- Running Step 7: Generating Plot ---")
            # We only need the one plot for the animation
            viz.plot_facies_maps(
                classification_results=pipeline_data['classification_results'],
                seismic_geometry=pipeline_data['raw_data']['seismic'],
                facies_names=config.FACIES_NAMES,
                output_most_likely=config.MOST_LIKELY_FACIES_MAP_PATH, # Overwrites this file
                output_grouped=config.GROUPED_FACIES_MAP_PATH # Overwrites this file
            )
            
            # 4. Copy the generated plot to the temp frame folder
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            shutil.copy(config.MOST_LIKELY_FACIES_MAP_PATH, frame_path)
            
            # --- 5. MODIFIED: Add text overlay to the frame ---
            print(f"  Adding text overlay to frame {i+1}...")
            try:
                # Get the text to draw
                text_overlay = _get_text_for_frame(param_to_animate, value)
                
                # Open the image file we just copied
                img = Image.open(frame_path)
                draw = ImageDraw.Draw(img)
                
                # Try to load a nice font, fall back to default
                try:
                    # You may need to change 'arial.ttf' to a font path on your system
                    # e.g., 'C:\\Windows\\Fonts\\arial.ttf' on Windows
                    # or '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf' on Linux
                    font = ImageFont.truetype("arial.ttf", size=24)
                except IOError:
                    print("  - Warning: Arial font not found, using default font.")
                    font = ImageFont.load_default()
                
                # Position for the text (top-left corner)
                pos = (10, 10)
                shadow_pos = (pos[0] + 2, pos[1] + 2)
                
                # Draw shadow (black)
                draw.text(shadow_pos, text_overlay, font=font, fill="black")
                # Draw main text (white)
                draw.text(pos, text_overlay, font=font, fill="white")
                
                # Save the modified image, overwriting the copy
                img.save(frame_path)
                
            except Exception as e:
                print(f"  - Warning: Could not add text overlay to {frame_path}. {e}")
            # --- End of text overlay modification ---
            
            frame_paths.append(frame_path)
            print(f"  Successfully generated and saved frame: {frame_path}")

        except Exception as e:
            print(f"!!! ERROR generating frame {i+1}: {e}")
            print("!!! Skipping this frame...")
            
    # --- Assemble GIF ---
    if not frame_paths:
        print("No frames were generated. Exiting.")
        return

    print("\n--- Assembling GIF ---")
    try:
        images = [Image.open(f) for f in frame_paths]
        
        safe_param_name = param_to_animate.replace('/', '_')
        
        # --- MODIFIED: Use global config vars for GIF name ---
        gif_name = f"animation_{safe_param_name}_min{ANIMATION_MIN}_max{ANIMATION_MAX}_step{ANIMATION_STEP}.gif"
        gif_path = os.path.join(final_dir, gif_name)
        
        images[0].save(
            gif_path, 
            save_all=True, 
            append_images=images[1:], 
            duration=frame_duration_ms, 
            loop=0
        )
        print(f"  Successfully created GIF: {gif_path}")
        
    except Exception as e:
        print(f"!!! ERROR creating GIF: {e}")

    # --- Clean up temp frames ---
    print("--- Cleaning up temporary frames ---")
    try:
        shutil.rmtree(temp_dir)
        print("  Temporary frames removed.")
    except Exception as e:
        print(f"  Error cleaning up temp folder: {e}")

    end_time = time.time()
    print(f"\n--- Animation Complete in {end_time - start_time:.2f} seconds ---")


# --- Main execution block ---
if __name__ == "__main__":
    
    # ======================================================================
    # --- CONFIGURE YOUR ANIMATION HERE ---
    # ======================================================================
    
    # 1. Choose ONE parameter to animate:
    # 'MC_SAMPLE_COUNT', 'AVO_THETA_ANGLES', 'CLASSIFICATION_METHOD'
    PARAM_TO_ANIMATE = 'MC_SAMPLE_COUNT'
    
    # 2. Define the range for the chosen parameter:
    # These values are used if param is 'MC_SAMPLE_COUNT' or 'AVO_THETA_ANGLES'
    ANIMATION_MIN = 0
    ANIMATION_MAX = 500
    ANIMATION_STEP = 20 # Use a large step, this process is slow!
    
    # 3. Define the duration of each frame in milliseconds
    FRAME_DURATION_MS = 700 
    
    # ======================================================================
    
    # --- This block automatically builds the list of values to iterate over ---
    animation_values_list = []
    
    if PARAM_TO_ANIMATE == 'MC_SAMPLE_COUNT':
        # Create a list of sample counts, e.g., [100, 1100, 2100, 3100, 4100]
        # We add +1 to max to make sure the max value is included if it's a multiple of step
        animation_values_list = np.arange(ANIMATION_MIN, ANIMATION_MAX + 1, ANIMATION_STEP)
    
    elif PARAM_TO_ANIMATE == 'AVO_THETA_ANGLES':
        # Create a list of angle arrays, where min/max/step define the MAX ANGLE
        # e.g., [np.arange(0,10,1), np.arange(0,20,1), np.arange(0,30,1)]
        ANGLE_STEP_DEGREE = 1 # This is the step *inside* the angle array, e.g., [0, 1, 2...]
        
        # Get the list of max angles, e.g., [10, 20, 30]
        max_angles = np.arange(ANIMATION_MIN, ANIMATION_MAX + 1, ANIMATION_STEP)
        
        # Create the list of angle arrays
        animation_values_list = [np.arange(0, max_angle, ANGLE_STEP_DEGREE) for max_angle in max_angles]

    elif PARAM_TO_ANIMATE == 'CLASSIFICATION_METHOD':
        # For this, min/max/step are ignored. Define the list manually.
        print(f"INFO: {PARAM_TO_ANIMATE} is selected. Ignoring MIN/MAX/STEP.")
        animation_values_list = ['bayesian'] # Add other methods if you implement them
        
    else:
        print(f"FATAL ERROR: Unknown PARAM_TO_ANIMATE: '{PARAM_TO_ANIMATE}'")
        print("Please choose 'MC_SAMPLE_COUNT', 'AVO_THETA_ANGLES', or 'CLASSIFICATION_METHOD'.")
        exit(1) # Exit the script
        
    if len(animation_values_list) == 0:
        print(f"FATAL ERROR: No animation values were generated for the given min/max/step.")
        print(f"MIN: {ANIMATION_MIN}, MAX: {ANIMATION_MAX}, STEP: {ANIMATION_STEP}")
        exit(1)
    
    # ======================================================================

    
    # Ensure the computed and output directories exist
    os.makedirs(config.COMPUTED_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Run the animation generator
    create_animation(
        param_to_animate=PARAM_TO_ANIMATE, 
        values=animation_values_list, 
        frame_duration_ms=FRAME_DURATION_MS
    )
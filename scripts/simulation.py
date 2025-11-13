"""
scripts/simulation.py

This module runs the full Monte Carlo and AVO simulation.
It replaces the old mc_draw.py.

CORRECTED VERSION: Matches 'mc_draw.py' logic exactly.
- Uses Facies IV samples as the variable top layer (instead of constant mean).
- Preserves the "Facies IV over Facies IV" offset logic.
"""

import numpy as np
import pickle

def avopp(vp1, vs1, d1, vp2, vs2, d2, ang, approx=1):
    """
    Calculates P-to-P reflectivity (Rpp) as a function of angle of incidence.
    Matches the original mc_draw.py / avopp.m exactly.
    """
    
    # Convert angles to radians
    t = ang * np.pi / 180.0
    p = np.sin(t) / vp1
    ct = np.cos(t)
    
    # Average and difference parameters
    da = (d1 + d2) / 2.0
    Dd = d2 - d1
    vpa = (vp1 + vp2) / 2.0
    Dvp = vp2 - vp1
    vsa = (vs1 + vs2) / 2.0
    Dvs = vs2 - vs1
    
    ip1 = vp1 * d1
    ip2 = vp2 * d2
    is1 = vs1 * d1
    is2 = vs2 * d2
    ipa = (ip1 + ip2) / 2.0
    Dip = ip2 - ip1
    isa = (is1 + is2) / 2.0
    Dis = is2 - is1
    
    if approx == 1:  # Full Zoeppritz (Aki & Richards)
        ct2 = np.sqrt(1 - (np.sin(t)**2 * (vp2**2 / vp1**2)))
        cj1 = np.sqrt(1 - (np.sin(t)**2 * (vs1**2 / vp1**2)))
        cj2 = np.sqrt(1 - (np.sin(t)**2 * (vs2**2 / vp1**2)))
        
        a = (d2 * (1 - 2 * vs2**2 * p**2)) - (d1 * (1 - 2 * vs1**2 * p**2))
        b = (d2 * (1 - 2 * vs2**2 * p**2)) + (2 * d1 * vs1**2 * p**2)
        c = (d1 * (1 - 2 * vs1**2 * p**2)) + (2 * d2 * vs2**2 * p**2)
        d = 2 * ((d2 * vs2**2) - (d1 * vs1**2))
        
        E = (b * ct / vp1) + (c * ct2 / vp2)
        F = (b * cj1 / vs1) + (c * cj2 / vs2)
        G = a - (d * ct * cj2 / (vp1 * vs2))
        H = a - (d * ct2 * cj1 / (vp2 * vs1))
        D = (E * F) + (G * H * p**2)
        
        Rpp = (((b * ct / vp1) - (c * ct2 / vp2)) * F - 
               (a + (d * ct * cj2 / (vp1 * vs2))) * H * p**2) / D
        
    elif approx == 2:  # Aki & Richards approximation
        Rpp = (0.5 * (1 - 4 * p**2 * vsa**2) * Dd / da + 
               Dvp / (2 * ct**2 * vpa) - 
               4 * p**2 * vsa * Dvs)
        
    elif approx == 3:  # Shuey
        poi1 = ((0.5 * (vp1 / vs1)**2) - 1) / ((vp1 / vs1)**2 - 1)
        poi2 = ((0.5 * (vp2 / vs2)**2) - 1) / ((vp2 / vs2)**2 - 1)
        poia = (poi1 + poi2) / 2.0
        Dpoi = poi2 - poi1
        
        Ro = 0.5 * ((Dvp / vpa) + (Dd / da))
        Bx = (Dvp / vpa) / ((Dvp / vpa) + (Dd / da))
        Ax = Bx - (2 * (1 + Bx) * (1 - 2 * poia) / (1 - poia))
        
        Rpp = (Ro + 
               ((Ax * Ro) + (Dpoi / (1 - poia)**2)) * np.sin(t)**2 + 
               0.5 * Dvp * (np.tan(t)**2 - np.sin(t)**2) / vpa)
        
    elif approx == 4:  # Shuey linear (Castagna)
        A = 0.5 * ((Dvp / vpa) + (Dd / da))
        B = (-2 * vsa**2 * Dd / (vpa**2 * da) + 
             0.5 * Dvp / vpa - 
             4 * vsa * Dvs / vpa**2)
        Rpp = A + B * np.sin(t)**2
        
    elif approx == 5:  # Wiggins 1983
        A = 0.5 * ((Dvp / vpa) + (Dd / da))
        C = 0.5 * (Dvp / vpa)
        B = C - 4 * (vsa / vpa)**2 * (Dvs / vsa) - 2 * (vsa / vpa)**2 * (Dd / da)
        Rpp = A + B * np.sin(t)**2 + C * np.tan(t)**2 * np.sin(t)**2
        
    elif approx == 6:  # Gidlow et al. 1992, Fatti et al. 1994
        a1 = 1 + np.tan(t)**2
        b1 = 8 * (vsa / vpa)**2 * np.sin(t)**2
        c1 = 0.5 * np.tan(t)**2 - 2 * (vsa / vpa)**2 * np.sin(t)**2
        Rpp = (a1 * Dip / (2 * ipa) - 
               b1 * Dis / (2 * isa) - 
               c1 * Dd / da)
    else:
        raise ValueError(f"Unknown approximation method: {approx}")
    
    return Rpp


def monte_carlo_draw(facies_stats, facies_names, n_draws):
    """
    Draws Monte Carlo samples from the facies distributions.
    """
    print(f"Drawing {n_draws} Monte Carlo samples for each facies...")
    print("="*60)
    
    mc_samples = {}
    
    for facies_name in facies_names:
        if facies_name not in facies_stats:
            print(f"  Warning: {facies_name} not in stats. Skipping.")
            continue
        
        # print(f"\nProcessing: {facies_name}")
        
        try:
            # Get the bivariate statistics for Vp and Vs
            mean_vector = facies_stats[facies_name]['bivariate_VpVs']['mean_vector']
            cov_matrix = facies_stats[facies_name]['bivariate_VpVs']['cov_matrix']
            
            # Get the univariate statistics for Density
            density_mean = facies_stats[facies_name]['univariate']['Density']['mean']
            density_std = facies_stats[facies_name]['univariate']['Density']['std']
        except KeyError as e:
            print(f"  ERROR: Missing stats for {facies_name}: {e}. Skipping.")
            continue
        
        # Check for valid statistics
        if np.any(np.isnan(mean_vector)) or np.any(np.isnan(cov_matrix)):
            print(f"  WARNING: Invalid Vp/Vs statistics. Skipping {facies_name}.")
            continue
            
        if np.isnan(density_mean) or np.isnan(density_std):
            print(f"  WARNING: Invalid Density statistics. Skipping {facies_name}.")
            continue
        
        # --- Draw Vp and Vs from BIVARIATE normal distribution ---
        try:
            vp_vs_samples = np.random.multivariate_normal(
                mean=mean_vector,
                cov=cov_matrix,
                size=n_draws
            )
            vp_samples = vp_vs_samples[:, 0]
            vs_samples = vp_vs_samples[:, 1]
            
        except np.linalg.LinAlgError as e:
            print(f"  ERROR: Covariance matrix is singular for {facies_name}.")
            print(f"  {e}")
            continue
        
        # --- Draw Density from UNIVARIATE normal distribution ---
        density_samples = np.random.normal(
            loc=density_mean,
            scale=density_std,
            size=n_draws
        )
        
        mc_samples[facies_name] = {
            'Vp': vp_samples,
            'Vs': vs_samples,
            'Density': density_samples
        }
        
        # print(f"  Vp:      mean={np.mean(vp_samples):.4f}, std={np.std(vp_samples):.4f}")
        
    print("\n" + "="*60)
    print(f"Monte Carlo sampling complete for {len(mc_samples)} facies.")
    print("="*60 + "\n")
    
    return mc_samples


def compute_avo_reflectivity(mc_samples, facies_names, theta_angles, approx=1):
    """
    Computes AVO reflectivity (Rpp) using Facies IV as the top layer.
    
    Args:
        mc_samples (dict): Output from monte_carlo_draw.
        facies_names (list): List of facies names.
        theta_angles (np.array): Array of angles (degrees).
        approx (int): Approximation flag for avopp.
    
    Returns:
        dict: A dictionary containing the computed AVO curves for each facies.
    """
    print("  Starting AVO reflectivity computation...")
    
    # --- FIX: Use Facies IV samples as top layer (from mc_draw.py logic) ---
    if 'FaciesIV' not in mc_samples:
        print("  ERROR: 'FaciesIV' not found in MC samples. Cannot run AVO.")
        return None

    facies_iv = mc_samples['FaciesIV']
    
    avo_data = {}
    
    # Loop through each facies (including FaciesIV itself)
    for facies_name in facies_names:
        if facies_name not in mc_samples:
            print(f"  Warning: {facies_name} not in MC samples. Skipping.")
            continue
        
        print(f"Processing AVO for: {facies_name}")
        
        samples = mc_samples[facies_name]
        n_sims = samples['Vp'].shape[0]
        n_angs = theta_angles.shape[0]
        
        # Initialize Rpp array
        rpp_curves = np.zeros((n_sims, n_angs))
        
        # Calculate reflectivity for each sample
        for i in range(n_sims):
            # --- Top Layer: Facies IV (Realization i) ---
            vp1 = facies_iv['Vp'][i]
            vs1 = facies_iv['Vs'][i]
            d1 = facies_iv['Density'][i]
            
            # --- Bottom Layer: Current Facies ---
            if facies_name == 'FaciesIV':
                # Special case: Offset index to ensure contrast (match mc_draw.py)
                j = (i + 1) % n_sims
                vp2 = samples['Vp'][j]
                vs2 = samples['Vs'][j]
                d2 = samples['Density'][j]
            else:
                vp2 = samples['Vp'][i]
                vs2 = samples['Vs'][i]
                d2 = samples['Density'][i]
            
            # Call avopp
            rpp_curves[i, :] = avopp(vp1, vs1, d1, vp2, vs2, d2, theta_angles, approx)
        
        avo_data[facies_name] = {
            'angles': theta_angles,
            'Rpp': rpp_curves
        }
    
    print("  AVO computation complete.")
    return avo_data


def run_avo_simulation(facies_stats, facies_names, well_data, n_samples, 
                         theta_angles, top_layer_depth_start, top_layer_depth_end, 
                         approx=1):
    """
    Runs the full simulation: MC draw -> AVO computation.
    
    Args:
        facies_stats (dict): Statistics for the 9 facies.
        facies_names (list): List of facies names.
        well_data (dict): The loaded Well2.txt data (NO LONGER USED for top layer).
        n_samples (int): Number of MC samples to draw.
        theta_angles (np.array): Array of angles (degrees).
        top_layer_depth_start (float): (Ignored - legacy).
        top_layer_depth_end (float): (Ignored - legacy).
        approx (int): Approximation flag for avopp.
    """
    print("="*60)
    print("  Starting Step 3: Monte Carlo & AVO Simulation (Corrected)")
    print(f"    MC Samples: {n_samples}")
    print(f"    Angle Range: {theta_angles.min()} to {theta_angles.max()} deg")
    print(f"    AVO approximation: {approx}")
    
    # Step 1: Monte Carlo sampling
    print("  Running Monte Carlo sampling...")
    mc_samples = monte_carlo_draw(facies_stats, facies_names, n_samples)
    
    if not mc_samples:
        print("  Error: No MC samples generated. Exiting.")
        return None
    
    # Step 2: Compute AVO reflectivity
    # We no longer need 'top_layer_props' because we use FaciesIV samples inside
    avo_data = compute_avo_reflectivity(
        mc_samples, 
        facies_names, 
        theta_angles, 
        approx
    )
    
    if not avo_data:
        print("  Error: AVO computation failed. Exiting.")
        return None
    
    return avo_data


def save_avo_data(avo_data, output_path):
    """
    Saves the synthetic AVO data (Rpp curves) to a .pkl file.
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(avo_data, f)
        print(f"Successfully saved synthetic AVO data to: {output_path}\n")
    except Exception as e:
        print(f"Error saving AVO data to {output_path}: {e}")


def load_avo_data(file_path):
    """
    Loads cached synthetic AVO data from a .pkl file.
    """
    print(f"Loading synthetic AVO data from: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Successfully loaded cached AVO data.")
        return data
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
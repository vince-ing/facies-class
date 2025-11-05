import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_statistics(filename='facies_statistics.pkl'):
    """
    Loads the pickled statistics file.
    
    Returns:
        dict: Statistics data containing facies_stats and plot_ranges
    """
    print(f"Loading statistics from: {filename}...")
    try:
        with open(filename, 'rb') as f:
            stats_data = pickle.load(f)
        print("  Statistics loaded successfully.\n")
        return stats_data
    except FileNotFoundError:
        print(f"  Error: Statistics file '{filename}' not found.")
        print("  Please run 'statistics.py' first.")
        return None
    except Exception as e:
        print(f"  Error loading {filename}: {e}")
        return None


def monte_carlo_draw(stats_data, n_draws=5000):
    """
    Draws Monte Carlo samples from the facies distributions.
    
    For each facies:
    - Draws Vp and Vs from a BIVARIATE normal distribution (correlated)
    - Draws Density from a UNIVARIATE normal distribution (independent)
    
    Args:
        stats_data (dict): The statistics dictionary from statistics.py
        n_draws (int): Number of Monte Carlo realizations per facies
        
    Returns:
        dict: Dictionary containing MC samples for each facies
              Format: {facies_name: {'Vp': array, 'Vs': array, 'Density': array}}
    """
    
    print(f"Drawing {n_draws} Monte Carlo samples for each facies...")
    print("="*60)
    
    mc_samples = {}
    
    for facies_name, facies_stats in stats_data['facies_stats'].items():
        
        print(f"\nProcessing: {facies_name}")
        
        # Get the bivariate statistics for Vp and Vs
        mean_vector = facies_stats['bivariate_VpVs']['mean_vector']
        cov_matrix = facies_stats['bivariate_VpVs']['cov_matrix']
        
        # Get the univariate statistics for Density
        density_mean = facies_stats['univariate']['Density']['mean']
        density_std = facies_stats['univariate']['Density']['std']
        
        # Check for valid statistics
        if np.any(np.isnan(mean_vector)) or np.any(np.isnan(cov_matrix)):
            print(f"  WARNING: Invalid Vp/Vs statistics. Skipping {facies_name}.")
            continue
            
        if np.isnan(density_mean) or np.isnan(density_std):
            print(f"  WARNING: Invalid Density statistics. Skipping {facies_name}.")
            continue
        
        # --- Draw Vp and Vs from BIVARIATE normal distribution ---
        try:
            # np.random.multivariate_normal returns shape (n_draws, 2)
            # Column 0 is Vp, Column 1 is Vs
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
        
        # Store the samples
        mc_samples[facies_name] = {
            'Vp': vp_samples,
            'Vs': vs_samples,
            'Density': density_samples
        }
        
        # Print verification statistics
        print(f"  Vp:      mean={np.mean(vp_samples):.4f}, std={np.std(vp_samples):.4f}")
        print(f"  Vs:      mean={np.mean(vs_samples):.4f}, std={np.std(vs_samples):.4f}")
        print(f"  Density: mean={np.mean(density_samples):.4f}, std={np.std(density_samples):.4f}")
        print(f"  Correlation(Vp,Vs): {np.corrcoef(vp_samples, vs_samples)[0,1]:.4f}")
    
    print("\n" + "="*60)
    print(f"Monte Carlo sampling complete for {len(mc_samples)} facies.")
    print("="*60 + "\n")
    
    return mc_samples


def avopp(vp1, vs1, d1, vp2, vs2, d2, ang, approx=1):
    """
    Calculates P-to-P reflectivity (Rpp) as a function of angle of incidence.
    
    Converted from MATLAB avopp.m function.
    
    Args:
        vp1, vs1, d1: P-wave velocity, S-wave velocity, density of layer 1 (top)
        vp2, vs2, d2: P-wave velocity, S-wave velocity, density of layer 2 (bottom)
        ang: array of angles in DEGREES
        approx: 1=Full Zoeppritz (default)
                2=Aki & Richards
                3=Shuey
                4=Shuey linear (Castagna)
                5=Wiggins 1983
                6=Gidlow et al. 1992
    
    Returns:
        Rpp: array of P-wave reflectivities at each angle
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


def compute_avo_reflectivity(mc_samples, angle_range=30, approx=1):
    """
    Computes AVO reflectivity curves for Facies IV overlying all facies.
    
    Args:
        mc_samples (dict): Dictionary of MC samples from monte_carlo_draw
        angle_range (float): Maximum angle in degrees (0 to angle_range)
        approx (int): Approximation method for avopp function
        
    Returns:
        dict: Dictionary containing reflectivity curves for each facies
              Format: {facies_name: {'angles': array, 'Rpp': 2D array (n_draws x n_angles)}}
    """
    
    print(f"\nComputing AVO reflectivity curves...")
    print(f"Angle range: 0° to {angle_range}°")
    print(f"Approximation method: {approx}")
    print("="*60)
    
    # Create angle vector
    angles = np.linspace(0, angle_range, 100)
    
    # Get Facies IV samples (this will be the upper layer for all calculations)
    if 'FaciesIV' not in mc_samples:
        print("ERROR: FaciesIV not found in MC samples!")
        return None
    
    facies_iv = mc_samples['FaciesIV']
    n_draws = len(facies_iv['Vp'])
    
    print(f"\nUsing FaciesIV as upper layer (n={n_draws} realizations)")
    print(f"Computing reflectivity for {len(angles)} angles\n")
    
    avo_data = {}
    
    # Loop through each facies (including FaciesIV itself)
    for facies_name, facies_samples in mc_samples.items():
        
        print(f"Processing: {facies_name}")
        
        # Initialize array to store all Rpp curves
        # Shape: (n_draws, n_angles)
        rpp_curves = np.zeros((n_draws, len(angles)))
        
        # Compute reflectivity for each MC realization
        for i in range(n_draws):
            # Upper layer (Facies IV)
            vp1 = facies_iv['Vp'][i]
            vs1 = facies_iv['Vs'][i]
            d1 = facies_iv['Density'][i]
            
            # Lower layer (current facies)
            # Special case: For FaciesIV, use a different realization to create variability
            if facies_name == 'FaciesIV':
                # Use the next realization (wrap around at the end)
                j = (i + 1) % n_draws
                vp2 = facies_samples['Vp'][j]
                vs2 = facies_samples['Vs'][j]
                d2 = facies_samples['Density'][j]
            else:
                vp2 = facies_samples['Vp'][i]
                vs2 = facies_samples['Vs'][i]
                d2 = facies_samples['Density'][i]
            
            # Calculate reflectivity
            rpp_curves[i, :] = avopp(vp1, vs1, d1, vp2, vs2, d2, angles, approx)
        
        # Store the results
        avo_data[facies_name] = {
            'angles': angles,
            'Rpp': rpp_curves
        }
        
        print(f"  Completed {n_draws} reflectivity curves")
    
    print("\n" + "="*60)
    print(f"AVO computation complete for {len(avo_data)} facies.")
    print("="*60 + "\n")
    
    return avo_data


def plot_avo_curves(avo_data):
    """
    Creates a 3x3 plot of Rpp vs Angle for all 9 facies.
    Shows all MC realizations as a cloud.
    
    Args:
        avo_data (dict): Dictionary from compute_avo_reflectivity
    """
    
    print("Creating AVO reflectivity diagnostic plots...")
    
    # Define the order and colors to match your reference image
    facies_order = [
        'FaciesIIaOil', 'FaciesIIbOil', 'FaciesIIcOil',
        'FaciesIIa', 'FaciesIIb', 'FaciesIIc',
        'FaciesIII', 'FaciesIV', 'FaciesV'
    ]
    
    facies_colors = {
        'FaciesIIaOil': 'darkred',
        'FaciesIIbOil': 'orange',
        'FaciesIIcOil': 'magenta',
        'FaciesIIa': 'darkred',
        'FaciesIIb': 'orange',
        'FaciesIIc': 'magenta',
        'FaciesIII': 'green',
        'FaciesIV': 'cyan',
        'FaciesV': 'blue'
    }
    
    # Create 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, facies_name in enumerate(facies_order):
        ax = axes[idx]
        
        if facies_name not in avo_data:
            ax.text(0.5, 0.5, f'{facies_name}\nNo Data', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 30)
            ax.set_ylim(-0.4, 0.3)
            continue
        
        angles = avo_data[facies_name]['angles']
        rpp_curves = avo_data[facies_name]['Rpp']
        color = facies_colors.get(facies_name, 'gray')
        
        # Plot all curves with transparency to show density
        for i in range(rpp_curves.shape[0]):
            ax.plot(angles, rpp_curves[i, :], 
                   color=color, alpha=0.90, linewidth=0.5)
        
        # Plot mean curve on top
        mean_curve = np.mean(rpp_curves, axis=0)
        ax.plot(angles, mean_curve, color='black', linewidth=2, 
               label='Mean', zorder=10)
        
        # Formatting
        ax.set_xlim(0, angles[-1])
        ax.set_ylim(-0.4, 0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Angle', fontsize=10)
        ax.set_ylabel('R$_{pp}$(θ)', fontsize=10)
        ax.set_title(f'{facies_name}', fontsize=12, fontweight='bold')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    fig.suptitle('Synthetic Angle-Dependent Reflectivity (Facies IV Overlying All Facies)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_filename = 'avo_reflectivity_curves.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"  Saved AVO plot to: {output_filename}")
    plt.close(fig)


def save_mc_samples(mc_samples, filename='mc_samples.pkl'):
    """
    Saves the Monte Carlo samples to a pickle file.
    
    Args:
        mc_samples (dict): Dictionary of MC samples
        filename (str): Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(mc_samples, f)
    print(f"Successfully saved MC samples to: {filename}\n")


def save_avo_data(avo_data, filename='avo_data.pkl'):
    """
    Saves the AVO reflectivity data to a pickle file.
    
    Args:
        avo_data (dict): Dictionary of AVO data
        filename (str): Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(avo_data, f)
    print(f"Successfully saved AVO data to: {filename}\n")


# --- Main execution ---
if __name__ == "__main__":
    
    # ============================================
    # USER PARAMETERS
    # ============================================
    N_DRAWS = 5000      # Number of Monte Carlo realizations per facies
    ANGLE_RANGE = 30    # Maximum angle in degrees (will be 0 to ANGLE_RANGE)
    APPROX = 1          # 1=Full Zoeppritz, 2=Aki&Richards, 3=Shuey, 4=Shuey linear
    # ============================================
    
    print("="*60)
    print("MONTE CARLO SAMPLING & AVO COMPUTATION")
    print("="*60)
    print(f"Number of draws per facies: {N_DRAWS}")
    print(f"Angle range for AVO: 0° to {ANGLE_RANGE}°")
    print(f"AVO approximation: {APPROX} (1=Zoeppritz)")
    print("="*60 + "\n")
    
    # 1. Load statistics
    stats_file = 'facies_statistics.pkl'
    all_stats = load_statistics(stats_file)
    
    if not all_stats:
        print("Error: Cannot proceed without statistics. Exiting.")
        exit()
    
    # 2. Perform Monte Carlo sampling
    mc_samples = monte_carlo_draw(all_stats, n_draws=N_DRAWS)
    
    if not mc_samples:
        print("Error: No MC samples generated. Exiting.")
        exit()
    
    # 3. Save the MC samples
    save_mc_samples(mc_samples, 'mc_samples.pkl')
    
    # 4. Compute AVO reflectivity curves
    avo_data = compute_avo_reflectivity(mc_samples, angle_range=ANGLE_RANGE, approx=APPROX)
    
    if not avo_data:
        print("Error: AVO computation failed. Exiting.")
        exit()
    
    # 5. Save AVO data
    save_avo_data(avo_data, 'avo_data.pkl')
    
    # 6. Create diagnostic plots
    plot_avo_curves(avo_data)
    
    # 7. Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total facies sampled: {len(mc_samples)}")
    print(f"Samples per facies: {N_DRAWS}")
    print(f"Angle range: 0° to {ANGLE_RANGE}°")
    print(f"AVO curves computed for: {len(avo_data)} facies")
    print("\nFiles created:")
    print("  - mc_samples.pkl (Monte Carlo samples)")
    print("  - avo_data.pkl (AVO reflectivity curves)")
    print("  - avo_reflectivity_curves.png (diagnostic plot)")
    print("\nNext step: Extract intercept and gradient from reflectivity curves")
    print("="*60)
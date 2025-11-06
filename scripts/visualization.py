"""
Core Responsibility: To generate and save all plots and maps for the analysis.

Conceptual Functions:

a) Log data plot with facies indicated

    Function: plot_well_log(well_data_df, output_path)

    Purpose: To plot the well log traces (Vp, Vs, etc.) alongside a block log of the facies.

    Input:

        well_data_df: The DataFrame from load_data.load_well_data(). This must contain 'Depth', 'Facies', and the log columns ('Vp', 'Vs', 'Density', etc.).

    Logic:

        Uses matplotlib.pyplot.subplots() to create a figure with multiple linked axes (e.g., 5 subplots: Vp, Vs, Density, Ip, Facies).

        For each log trace (Vp, Vs, etc.), it plots trace vs. Depth.

        For the 'Facies' column, it creates a "block log" (e.g., using plt.pcolormesh) where the color of each block corresponds to the facies at that depth.

        Links all Y-axes (Depth) so they scroll together.

        Saves the figure to output_path.

b) Plots of PDFs and CDFs for all classes

    Function: plot_property_distributions(well_data_df, properties, facies_names, output_path_prefix)

    Purpose: To plot the PDF (Probability Density Function) and CDF (Cumulative Distribution Function) for various properties, separated by facies.

    Input:

        well_data_df: The well log DataFrame, which must contain the 'Facies' column.

        properties: A list of property names to plot (e.g., ['Vp', 'Vs', 'Density', 'Ip', 'VpVs']). main.py would be responsible for calculating 'Ip' and 'VpVs' and adding them to the DataFrame before calling this.

        facies_names: The list of facies to plot.

    Logic:

        Loops through each prop in the properties list:

            Creates a figure for the PDF, and another for the CDF.

            Loops through each facies_name in facies_names:

                Extracts the data for that facies: data = well_data_df[well_data_df['Facies'] == facies_name][prop]

                PDF Plot: Uses seaborn.kdeplot(data, label=facies_name) to plot the smoothed probability density.

                CDF Plot: Sorts the data and plots it as a cumulative line graph (x = np.sort(data), y = np.arange(1, len(x)+1) / len(x)).

            Saves the PDF plot to f"{output_path_prefix}_{prop}_pdf.png" and the CDF plot to f"{output_path_prefix}_{prop}_cdf.png".

c) Reflectivity curves as a function of angle

    Function: plot_avo_curves(avo_data, output_path)

    Purpose: To plot the calculated AVO reflectivity curves for all facies.

    Input:

        avo_data: The data structure from feature_engineering.load_avo_data(). This is likely a dictionary like {'FaciesIIa': {'angle': array, 'reflectivity': array}, ...}.

    Logic:

        Creates one plot.

        Loops through each facies_name, data in avo_data.items().

        Plots data['angle'] vs. data['reflectivity'] and gives it a label.

        Adds a legend, title, and axis labels.

        Saves to output_path.

d) Intercept-Gradient cross plots

    Function: plot_ig_crossplot(seismic_ig_data, mc_samples_data, facies_names, output_path)

    Purpose: To visualize the 2D distributions of the simulated facies and the real seismic data.

    Input:

        seismic_ig_data: The (N, 2) array of real seismic data (from feature_engineering.load_ig_features()).

        mc_samples_data: The {'samples': ..., 'labels': ...} dictionary (from simulation.load_mc_samples()).

        facies_names: List of facies names for labeling.

    Logic:

        Uses seaborn.jointplot or seaborn.kdeplot for rich visualization.

        Plots the seismic_ig_data as a 2D density plot (e.g., in grayscale) to show the "background" of real data.

        Overlays on the same plot a 2D kdeplot for each facies from the mc_samples_data, using a different color for each.

        Saves to output_path.

e) Image of most-likely facies & f) Image of most-likely grouped facies

    Function: plot_facies_map(map_data_1d, original_shape, tick_labels, cmap, output_path)

    Purpose: A flexible function to plot any 2D facies map. We use one function for both (e) and (f).

    Input:

        map_data_1d: A 1D array of integer indices (e.g., [0, 1, 2, 0, ...] where 0='FaciesIIa', 1='FaciesIIaOil', etc.).

        original_shape: The 2D shape of the original seismic map (e.g., (200, 300)).

        tick_labels: A list of string labels for the colorbar (e.g., ['FaciesIIa', ...] OR ['Oil', 'Brine', 'Shale']).

        cmap: The matplotlib colormap to use.

    Logic:

        Reshapes the 1D data into a 2D map: map_2d = map_data_1d.reshape(original_shape).

        Uses plt.imshow(map_2d, cmap=cmap).

        Creates a plt.colorbar() and sets its tick labels to tick_labels.

        Turns off the axes (plt.axis('off')).

        Saves to output_path.

    How main.py would use this for (e) and (f):

        For (e): plot_facies_map(results['most_likely_map'], shape, config.FACIES_NAMES, config.FACIES_CMAP, ...)

        For (f): main.py would first compute the grouped data, get its unique labels and integer indices, and then call: plot_facies_map(grouped_indices, shape, grouped_labels, config.GROUPED_CMAP, ...)
"""
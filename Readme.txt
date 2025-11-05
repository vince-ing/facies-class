Well2 txt file has the well log data for all the full depth range. Each curve is in a column, and the labels are at the top. 

Facies.zip contains 9 txt files, one for each facies. Each has a series of well curves in each column. The depths are for each individual facies. In other words, these have been labelled from Well 2 data and then pulled out for convenience.

Seismic.zip contains 4 txt files:
Inline numbers (1-D vector, 245x1)
Cross line numbers (1-D vector, 506x1)
Intercept (2D matrix 245x506)
Gradient (2D matrix 245x506)

Intercept and Gradient matrices have a few values that are NaN. Depending on the language you use, you'll need to change that to the appropriate indicator.


Mineral&FluidProperties lists those used in fluid substitution. You won't need these but are for references.

avopp.m is the Matlab function that computes reflectivity as a function of angle, Vp, Vs, and density for 2 layers. You'll need to convert this to another language if appropriate.

BlockedGray.m is Matlab function for a colormap used to plot the maps of grouped most likely facies.
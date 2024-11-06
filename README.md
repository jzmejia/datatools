# datatools
A suite of tools for data processing, plotting, and exploration

**gps_tools.py** - Import GAMIT/GLOBK post-process GNSS station positions to python. Specifically for glaciological applications using the OnIce class that will preform coordinate transformations to along-flow/across-flow directions, calculate ice velocities, etc. 

**diurnal.py** - Find minimuma and maxima extrema values for diurnally varying timeseries data. 

[![DOI](https://zenodo.org/badge/511994691.svg)](https://zenodo.org/badge/latestdoi/511994691)  Archival, July 2022, for script assocaited with publication [Mejia et al., 2022](https://doi.org/10.1029/2022GL100058).  

Features:

- PredictExtrema: allows you to enter in a daterange of good data, the extrema picks during this timeperiod will be the center of windows when picking extrema for the entire timeseries. 
- Does not cut off picks for an arbitrary midnight value, using a 24-hour window that encompasses minimum and maximum picks. 
- Allows you to require either the minimua or maxima occur first.
  
**melt_model.py** - Read in automatic weather station data and prepare for the implements the enhanced-temperature index model by [Pellicciotti et al., 2005](https://doi.org/10.3189/172756505781829124).

- Calculate daily albedo values during solar noon
- Adjust incoming or reflected solar radiation data for repeated shadowing of sensors
- Calculate hourly melt rates using the enhanced temperature-index model Future functionality: remove restrictions on input data (allow implementation as a function)
- Script implementation example with figures (jupyter notebook) can be found here [Mejia et al., 2021](https://doi.org/10.18739/A2V97ZS6G).

**hydrotools.py** - Calculate stream stage and hydraulic head from piezometer and water level sensor measurements.
  
 
These scripts have been developed to aid in data analysis throughout my dissertation work studying how meltwater impacts the ice dyanmics of the Greenland Ice Sheet. The class `DiurnalExtrema` has specific inputs to customize and predict when diurnal extrema will occur so that noise in the timeseries data set has a minimial impact on extrema picks.

All scripts are actively being refined and improved. Updates will add functionality.

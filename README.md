# datatools
A suite of tools for data processing, plotting, and exploration

**gps_tools.py** - Import GAMIT/GLOBK post-process GNSS station positions to python. Specifically for glaciological applications using the OnIce class that will preform coordinate transformations to along-flow/across-flow directions, calculate ice velocities, etc. 

**diurnal.py** - Find minimuma and maxima extrema values for diurnally varying timeseries data.
Features:
  - PredictExtrema: allows you to enter in a daterange of good data, the extrema picks during this timeperiod will be the center of windows when picking extrema for the entire timeseries. 
  - Does not cut off picks for an arbitrary midnight value, using a 24-hour window that encompasses minimum and maximum picks. 
  - Allows you to require either the minimua or maxima occur first.
  
  
 
These scripts have been developed to aid in data analysis throughout my dissertation work studying how meltwater impacts the ice dyanmics of the Greenland Ice Sheet. The class `DiurnalExtrema` has specific inputs to customize and predict when diurnal extrema will occur so that noise in the timeseries data set has a minimial impact on extrema picks.

All scripts are actively being refined and improved. Updates will add functionality.

#!/user/bin/env python

"""
gpstools.py
Created by: JZMejia

A collection of tools for reading in and working with GNSS data. 
Tool development primarily for cryospheric applications but many
are universal.

Last update: 24 Oct 2023
 """

#----------------------------------------------------------------------|------|

from math import atan, sin, cos, sqrt
# from datetime import timedelta
from pathlib import PurePath, Path
import time
from typing import List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from scipy import signal


from utils import c_rolling
from diurnal import to_exact_indexing
# <TODO: automatically see which columns are in file & import >

FrameOrSeries = Union[pd.DataFrame, pd.Series]
WindowTypes = Union[Tuple[str], Tuple[str, str]]
WindowList = List[WindowTypes]

SECONDS_PER_DAY = 60*60*24


# Dictionaries
# fill dics with station sepcific information
# station_names, (tuple(str, str))
# optional - for convenience

# station_names = (
#     # ('receiver code (4 characters)','location code (4 chars')
#     ('USF1', 'LMID'),
#     ('USFK', 'ROCK'),
#     ('USFS', 'SBPI')
# )

# EGM2008 = {
#     # station locationi code: EGM2008 elevation (m)
#     'lmid': 29.0059,
#     'kaga': 28.3183,
#     'rock': 27.1274
# }
# EGM96 = {
#     # station location code: EGM96 elevation (m)
#     'lmid': 29.2909,
#     'kaga': 29.0784,
#     'rock': 27.6441
# }


# # background velocities for stations - optional used with onice stations
# background_velocities_mpd = (
#     # (station location code, background velocity in meters per day)
#     ('LMID', 0.244),
# )


_directions = (
    ('n', 'dnorth'),
    ('e', 'deast'),
    ('u', 'dheight'),
    ('x', 'xflow'),
    ('t', 'xtran'),
)


class BaseStn:
    """Base station (static) info to adjust processed station positions.

    Define base station coordinates for reference station position 
    determination and transformation. All coordinates should be defined
    using GAMIT/GLOBK position files as the same naming conventions are
    used. 

    Cartesian coordinate system "site_pos"
        X = (N + h) * cos(phi) * cos(lambda)
        Y = (N + h) * cos(phi) * sin(lambda)
        Z = ((1 - e ** 2) * N + h) * sin(phi)
        where: 
            e ** 2 = 2 * flatttening - flattening ** 2
            East-West radius of curvature
            N ** 2 = a / (1 - e * sin(phi) ** 2)



    Parameters
    ----------
        name (str): 4-char station name
        site_pos (tuple): Cartesian station coordinates
            (X, Y, Z) 
        geod_pos (tuple): Geodetic coordinates
            (Geodetic co-latitude, longitue, elipsoidal height)

    Attributes
    ----------
        .loc_coord (tuple): Local coordinates (m)
            NEU - Northing, Easting, Vertical
        .Lat (float): Geodetic co-latitude (degrees)
        .Long (float): Geodetic longitude (-180 to 180 degrees)
        .ellipsoidal_height (float): (m)
        .X (float): unit (m) 
        .Y (float): 
        .Z (float): 
        
        TODO: make calculation to get lat/lon from XYZ coords.
    """

    def __init__(self,
                 name: str,
                 site_pos: Tuple[float, float, float],
                 geod_pos: Tuple[float, float, float]
                 ):
        self.site_pos = site_pos
        self.Lat, self.Long, self.ellipsoidal_height = geod_pos
        self.Long = (360 - self.Long)*-1 if self.Long > 180 else self.Long
        self.geod_pos = (self.Lat, self.Long, self.ellipsoidal_height)
        self.X, self.Y, self.Z = site_pos
        self.name = name.upper()
    

    def __str__(self):
        return 'Base Station Object'

    def __repr__(self):
        return 'Base Station '+self.name
    
    def geoid_height(self):
        pass
        # [value for key, value in EGM2008.items() 
        #  if key.upper() == self.name][0]
    
    def elevation(self,geoid_height):
        return self.ellipsoidal_height - geoid_height


# define base stations used in deployment
# (enables GPS elevation calculations)
# e.g.
# rock = BaseStn('ROCK',
#                site_pos=(1412215.2584, -1711212.5767, 5960386.7316),
#                geod_pos=(69.708219352, 309.531891746, 594.5942))

# kaga = BaseStn('KAGA',
#                site_pos=(1464296.0967, -1733658.2881, 5940955.0164),
#                geod_pos=(69.222301946, 310.185368004, 150.0098))

# defined_base_stations = {
#     'ROCK': rock,
#     'KAGA': kaga
# }

# dictionary of antenna adjustments
# adjustment (positive is raising the antenna, unit meters)
# example
# antenna_adjustments = {
#     'LMID': {
#         'date': pd.Timestamp('2018-07-07 13:28:00'),
#         'adjustment': -2,
#         'drop_until': pd.Timestamp('2018-07-07 21:40:00')
#     },
#     'JEME': {
#         'date': pd.Timestamp('2018-07-17 23:40:00'),
#         'adjustment': -2,
#         'drop_until': pd.Timestamp('2018-07-18 00:45:00')
#     }
# }


def get_station_name(gps_data: Union[Path, PurePath, str, FrameOrSeries],
                     **kwargs) -> Tuple[str, str]:
    """return station ID and name from id in file name

    Args:
        gps_data (Path): path to gps data with file in MoVE format
            which has the station ID in the file name. 

    Returns:
        Tuple[str, str]: station's ID and Name (4 char str)
    """

    if isinstance(gps_data, (Path, PurePath, str)):
        stn_ID = PurePath(gps_data).name[:4].upper()
    else:
        stn_ID = kwargs['stn_ID'] if 'stn_ID' in kwargs else input(
            'station ID(gnss receiver name), e.g. "usf1"')
    return stn_ID, stn_ID


class OnIce:
    """
    Read in on ice GNSS station NEU position files and project to flow direction.

    Arguments
    ---------
        file: str, pd.DataFrame
        default file naming conventions preferred
        './data_directory/NAME_YR_DTYPE.txt'
        first four chars of the file are the station ID

    Attributes
    ----------
        data (pandas.DataFrame) : Data from file in a pandas data frame.
        file (str) : Path to datafile
        stn_ID (str) : 4 character station ID determined from input file name.
        stn (str) : 4 character station name.
            determed from station_names.py
            if no output is 'ukwn' - unknown
            check/update station_names.py
        dnorth/.deast: series
            datetime indexed
            northing (m)/easting (m)
            from base station

    Methods
    -------
        Create an object using GPSTools
        obj1 = GPSTools(dnorth, deast, uplift)

        return the value for transform_to_xflow
        obj1.transform_to_xflow()
    """

    def __init__(self,
                 gps_data: Union[Path, PurePath, str, FrameOrSeries],
                 base_stn: Optional[str] = None,
                 **kwargs):

        self.file_name = Path(gps_data) if isinstance(
            gps_data, (Path, PurePath, str)) else None
        self.data = _get_data(gps_data)
        self.stn_ID, self.stn = get_station_name(gps_data, **kwargs)
        self.date = self.data.index
        self.doy = self.data.doy
        self.year = self.date[0].year if self.date.year[0] == self.date.year[-1] else None
        self.base_stn = base_stn
        # * Change assigning attrs from explicit to implicit using a dic
        # * with a list of col names

        self.dnorth = self.data.dnorth
        self.deast = self.data.deast

        self.z = self.data['dheight']
        # self.ellipsoidal_height = self._calc_ellipsoidal_height()
        # self.elevation = self._calc_elevation() if self.ellipsoidal_height 
        # is not None else None

        if 'dnorth_err' in self.data:
            self.errs = pd.DataFrame(
                {'N': self.data.dnorth_err, 'E': self.data.deast_err,
                    'U': self.data.dheight_err})
        self.sampling_rate = infer_sampling(self.data)
        
        # first pass at reprojection to station flow direction
                # if _is_file(gps_data):
        #     else:
        # self.xflow = self.data['xflow']
        # self.xtran = self.data['xtran']
        self.xflow, self.xtran = self.reproject_to_flow_direction()
        
        # self.dist_from_basestn = self.calc_dist(self.data.dnorth,self.data.deast)
        # self.data['Dist'] = self.dist_from_basestn

        # self.baseline = self.dist_from_basestn.dropna()[0]
    
        self.vel_header = None
        
    def calc_dist(a,b):
        """distance between a and b"""
        return sqrt(a**2+b**2)
    
    
    def reproject_to_flow_direction(self, alpha=None, update_instance=True, 
                                    preserve_ref=False):
        """reproject northing and easting positions by alpha radians
        
        direction_diff is shifted to the origin (data - data at t=0)
        this means each timeseries will start at 0 (origin)
        
        Parameters
        ----------
        alpha : float, optional
            angle alpha in radians used to transform coordinate system, 
            by default None. If no alpha is given alpha will be calculated
            using the first and last positions defined in self.data.dnorth
            and self.data.deast
        update_instance : bool, optional
            update instance with calculated reprojection, by default True
        preserve_ref : bool, optional
            preserve reference frame of position data, by default False. 
            In most cases positions are defined as distance from a reference
            point (e.g., a base station's location). If reference is 
            preserved xflow and xtran will be calculated as distance from
            this reference point in the along flow and across flow 
            directions. Otherwise, xflow and xtran will be calculated 
            as distance in either direction from the starting position 
            at time t=0.
            
        Returns
        -------
        xflow : pd.Series
            Position timeseries in the along-flow direction defined by alpha.
            Units are same as input data. 
        xtran : pd.Series
            Position timeseries in the across-flow direction.
            xtran defined as distance from starting position at time t=0
            in the direction perpendicular to flow. Units are preserved.
        """

        if preserve_ref:
            dnorth, deast = self.dnorth, self.deast
        else:
            t0 = self.determine_start()
            # calc xflow and xtran as dist from starting position at time t0
            dnorth = self.dnorth - self.dnorth.dropna()[t0]
            deast =  self.deast - self.deast.dropna()[t0]
            # self.horizontal_disp = self.calc_dist(dnorth,deast)

        
        alpha = alpha if alpha else self.flow_direction_angle()
        xflow = self.calc_xflow(alpha,deast,dnorth)
        xtran = self.calc_xtran(alpha,deast,dnorth)

        # update data storage dataframe by adding new columns
        if update_instance:
            self.data['xtran'] = xtran
            self.data['xflow'] = xflow
            self.xflow = xflow
            self.xtran = xtran
            self.alpha = alpha
            
        return xflow, xtran
    
    def flow_direction_angle(self):
        """calculate alpha from first and last positions"""
        dnorth_daily = self.dnorth.resample('1D').mean().dropna()
        deast_daily = self.deast.resample('1D').mean().dropna()
        
        alpha = atan((dnorth_daily[-1] - dnorth_daily[0])
                            / (deast_daily[-1] - deast_daily[0]))
        
        return alpha
        

    def determine_start(self):
        idx=0
        for i in range(100):
            diff=self.data.index[idx+1]-self.data.index[idx]
            if diff.seconds <= self.sampling_rate:
                break
        return i

    def __str__(self):
        return print('gps data')
    
    def calc_xflow(self,alpha,easting,northing):
        """transform northing and easting to along flow reference frame"""
        return cos(alpha)*easting + sin(alpha)*northing
    
    def calc_xtran(self,alpha,easting,northing):
        """transform northing and easting to across flow reference frame"""
        return -1*sin(alpha)*easting + cos(alpha)*northing

    # def _antenna_lowering_correction(self, correct_years: list) -> pd.Series:
    #     """correct vertical gps position for antenna adjustments."""
    #     if self.stn in antenna_adjustments and self.year in correct_years:
    #         info = antenna_adjustments[self.stn]
    #         adjust_at = info['date']

    #         z_adj = self.data['dheight'][adjust_at:]-info['adjustment']
    #         # modify dataframe self.data with adjusted height
    #         self.data.loc[z_adj.index, 'dheight'] = z_adj

    #         if info['drop_until'] is not None:
    #             self.data = self.data.drop(
    #                 self.data[adjust_at:info['drop_until']].index)
    #     return self.data[['dheight']]

    def plot_NEU(self, **kwargs):
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                               constrained_layout=True)
        ax[0].plot(self.dnorth, '.', **kwargs)
        ax[0].set_ylabel('Northing (m)')
        ax[1].plot(self.deast, '.', **kwargs)
        ax[1].set_ylabel('Easting (m)')
        ax[2].plot(self.z, '.', **kwargs)
        ax[2].set_ylabel('Height (m)')
        pass

    def plot(self, **kwargs):
        plt.figure()
        plt.plot(self.xflow, '.', **kwargs)
        plt.plot(c_rolling(self.xflow, '6H',
                           min_periods=self.samples_in_timespan('2H')))
        pass

    def drop_positions(self, droplist: list, apply_to_instance=False):
        dropDF = _subset_from_windows(self.data, droplist)
        if apply_to_instance:
            self.drop_from_instance(dropDF.index)
        return dropDF

    # def _calc_ellipsoidal_height(self) -> Union[FrameOrSeries, None]:
    #     """base-onIce height difference to ellipsoidal height in m"""
    #     ellip_hgt = None
    #     if self.base_stn is not None:
    #         ellip_hgt = self.base_stn.ellipsoidal_height + self.data.dheight
    #         ellip_hgt = ellip_hgt.rename('ellipsoidal_height')
    #     return ellip_hgt

    # def _calc_elevation(self):
    #     local_geoid_height = [val for stn, val in EGM2008.items()
    #                           if stn.upper() == self.stn][0]
    #     elev = self.ellipsoidal_height - local_geoid_height
    #     return elev.rename('elevation')

    def drop_from_instance(self, indices):
        self.data = self.data.drop(indices)
        self.xflow = self.xflow.drop(indices)
        self.xtran = self.xtran.drop(indices)
        self.dnorth = self.dnorth.drop(indices)
        self.deast = self.deast.drop(indices)
        self.z = self.z.drop(indices)
        if self.elevation is not None:
            self.elevation = self.elevation.drop(indices)
            self.ellipsoidal_height = self.ellipsoidal_height.drop(indices)
        pass

    # functions to calculate along flow direction

    def _is_positive(self, direction):
        return self.data[direction].gt(0).all()

    def _which_quadrent(self):
        """quadrent of dnorth, deast, with base station at origin. """
        n_quad = [1, 2] if self._is_positive('dnorth') else [3, 4]
        e_quad = [1, 4] if self._is_positive('deast') else [2, 3]
        return [x for x in e_quad if x in n_quad][0]

    # def _get_NE_bounds(self):
    #     d = self.data[['dnorth', 'deast']].resample('1D').mean().dropna()
    #     return (d.dnorth[0], d.dnorth[-1]), (d.deast[0], d.deast[-1])

    # def _get_NE_range(self):
    #     dN, dE = _get_NE_bounds()
    #     return dN[1]-dN[0], dE[1]-dE[0]

    def linear_detrend(self,
                       component: str,
                       from_dataframe: Optional[FrameOrSeries] = None,
                       window: Optional[WindowTypes] = None,
                       update_dataDF=True,
                       ):
        """
        return linearly detrended timeseries GPS data

        Parameters
        ----------
            component (str) : column names of direction data to detrend.
                options are self.data.columns strings
            from_dataframe (pd.DataFrame) : Defaults to self.data.
                Dataframe to call data from
            window : tuple, list, int, pd.Period
                start and end of timeseries to be detrended
                valid input types are accepted by
                datatools.diurnal function to_exact_indexing()
            update_dataDF (bool) : Defaults to True.

        Returns
        -------
            detrended (series) : linearly detrended timeseries
        """
        data = self.data if from_dataframe is None else from_dataframe
        df = clip_to_window(data, window, col_name=['doy', component]
                            ).dropna(how='any')
        # TODO: change to use numpy's Polynomial.polyfit()
        pfit = np.polyfit(df['doy'], df[component], 1)
        detrended = data[component] - pfit[1] - pfit[0] * data['doy']
        self.detrended = detrended
        if update_dataDF:
            self.data[component + '_detrended'] = detrended
        return detrended

    def z_detrend_wrt_xflow(self,
                            window: tuple,
                            gen_figure=False):
        """Detrend vertical position data in the along-flow direction.

        Args:
            window (tuple): timespan to detrend z with
            gen_figure (bool, optional): Defaults to False.

        Returns:
            df (pandas.DataFrame): detrended vertical position timeseries.
        """
        df = pd.DataFrame({'z': self.z, 'xflow': self.xflow,
                           'timestamp': self.z.index})
        df = clip_to_window(df, window, col_name='z')
        # switch index to along flow direction (will be detrended wrt index)
        df = df.set_index('xflow')
        
        df['dheight_xdetrended'] = signal.detrend(df['z'], type='linear')
        df = df.set_index('timestamp')
        return df

    def samples_in_timespan(self, timespan: str) -> int:
        """int number of measurements in a given pandas readable timespan"""
        return round(pd.Timedelta(timespan).total_seconds()/self.sampling_rate)

    def calc_velocity(self,
                      component: str,
                      stat_window='3T',
                      separation_window='2H',
                      smoothing: Optional[str] = None,
                      set_min_periods: Optional[Union[int, None]] = None,
                      window: Optional[WindowTypes] = None,
                      timeit=False
                      ) -> pd.DataFrame:
        """calculate velocity from position timeseries

        Args:
            component (str): string containing letters of the
                directions to calcualte for.
                'n' northing
                'e' easting
                'u' vertical
                'x' along-flow direction
                't' transverse-flow direction
            stat_window (str, optional): pandas readable time string 
                for position mean. Defaults to 3 minutes.
            separation_window (str, optional): number of hours between
                positions measurements to determine displacement.
                Defaults to 2 hours.
            smoothing (int, optional):
                Rolling mean window length applied to positions before
                calculating velocity. No smoothing applied if None.
                Defaults to None.
            set_min_periods (int, optional):
                minimum number of observations required in applied 
                smoothing of position data before velocity calculation.
                Note, if value given to set_min_periods is greater than
                the number of observations in that timespan there will be
                no datapoints and an empty timeseries. Defaults to None.
            window (tuple) : timespan to perform velocity calculation.
                Defaults to entire timeseries.
            timeit (bool, optional) : print the runtime to screen.
                Defaults to False.

        Returns:
            vel_df (pd.DataFrame): time-indexed dataframe, columns with
                velocities for each timestamp and component of motion.
        """

        coord_labels = find_label(component)
        t_shift = pd.Timedelta(separation_window) / 2
        stat_shift = pd.Timedelta(stat_window) / 2
        dt = pct_day(separation_window)
        all_velocities = []

        df = clip_to_window(self.data, window, coord_labels[0])[coord_labels]

        if smoothing is not None:
            min_periods = self.samples_in_timespan(
                smoothing) if set_min_periods is False else set_min_periods
            df = c_rolling(df, smoothing, min_periods=min_periods)

        # create a series of datetimes in intervals=stat_window
        binned_timeseries = pd.date_range(
            start=df.index[0] + pd.Timedelta(separation_window),
            end=df.index[-1] - pd.Timedelta(separation_window)
            + pd.Timedelta('0.1s'), freq=stat_window)

        t = time.time() if timeit else False
        for idx in binned_timeseries:
            df0, df1 = position_subsets(
                df, idx, t_shift, stat_shift, closed='left')
            if not is_good(df0, df1, 3):
                continue
            velocities = [idx]
            for label in coord_labels:
                velocities.append(vel_equ(df0, df1, label, dt))
            all_velocities.append(velocities)

        cols = ['date']
        # * Currently columns are named according to the short hand
        # * directions typed.
        # TODO create an option to define your own column names
        cols.extend([a for a, b in _directions if a in component])
        self.vel = pd.DataFrame(all_velocities, columns=cols).set_index('date')
        runtime(t)

        # populate processing dic for header
        # _create_vel_header(component, stat_window, separation_window, window)
        return self.vel

    # def _create_vel_header(component,
    #                        stat_window,
    #                        separation_window,
    #                        window):
    #     header = ''
    # pass

    def _name_file(self, DAT: str, FLAG: str, ext='.csv') -> str:
        """generates file name in the format CODEYY_DAT_FLAG.ext

        Args:
            DAT (str): data type, capital letters, 
                VEL velocity data,
                ROV for roving dGPS, 
                GPS for positions,
                UPL for uplift
            FLAG (str): speical conditions/subtypes identifier.
                NEUXT (components of motion Northing, easting, etc.)
                RAW 
                W Warning
                A All data
            ext (str, optional): [description]. Defaults to '.csv'.

        Returns:
            file_name (str): e.g. LMID17_VEL_NEUXT.csv
        """
        CODEYY = self.stn.upper()+str(self.year)[2:]+'_'
        return CODEYY+DAT.upper()+'_'+FLAG.upper()+ext


# FUNCTIONS

def _is_file(obj):
    return isinstance(obj, (Path, PurePath, str))


def _get_data(gps_data: Union[Path, PurePath, str, FrameOrSeries]):
    return gps_data if isinstance(gps_data, pd.DataFrame
                                  ) else load_NEUgps(gps_data)


def runtime(t):
    if t:
        elapsed = time.time() - t
        print('Elapsed Time:')
        if elapsed > 60:
            minutes = int(elapsed // 60)
            seconds = elapsed - (minutes * 60)
            print(f'{minutes:02.0f}:{seconds:02.0f}')
        else:
            print(f'{elapsed:02.0f} seconds')
    pass


def drop_large_errors(df: FrameOrSeries,
                      threshold: Optional[float] = None):
    """drop errors above threshold from dataframe"""
    if threshold:
        df = df[df.dnorth_err < threshold]
        df = df[df.deast_err < threshold]
    return df


def filter_zero(df):
    return df[df.X_vel <= 0].index


# def calc2Ddist(stn1, stn2, idx):
#     '''Calculate the 2D (horizontal) distance between
#     two GPS stations from dNorth (m) and dEast (m) locations
#     for a given time (index) '''

#     return


# loading data functions


def load_NEUgps(file):
    col_names = ['doy', 'dnorth', 'dnorth_err', 'deast',
                 'deast_err', 'dheight', 'dheight_err', 'err_code']
    df = pd.read_csv(file, index_col=0, parse_dates=True, names=col_names,
                     dtype={'index': '<M8[ns]', 'doy': np.float64,
                            'dnorth_err': np.float64, 'deast': np.float64,
                            'deast_err': np.float64, 'dheight': np.float64,
                            'dheight_err': np.float64, 'err_code': np.int64},
                     na_values='  nan')
    df.drop(columns='err_code', inplace=True)
    df.tz_localize('UTC')
    return df


def load_NEUXTvel(file):
    '''
    Load calculated NEUXT velocity csv files for MoVE GPS stns.
    Parameters
    ----------
        file: str
             .csv from ./data_csv.
    Outputs
    -------
        df: pd.DataFrame
            units: meters per day
            df.N_vel - North velocity velocity (S negative)
            df.E_vel - East velocity (W negative)
            df.U_vel - Vertical velocity (down negative)
            df.X_vel - Along flow velocity
            df.T_vel - Transverse to flow velocity
            Units: meters per day

    '''
    df = pd.read_csv(file, index_col=0, parse_dates=True, na_values="  nan",
                     dtype={'N_vel': np.float64, 'E_vel': np.float64,
                            'U_vel': np.float64, 'X_vel': np.float64,
                            'T_vel': np.float64})
    df.tz_localize('UTC')
    return df


def subset_from_bounds(series, bounds, closed):
    """use bounds to slice the pandas series or df, return subset

    closed : str, default None
    Make the interval closed on the ‘right’, ‘left’, ‘both’ or ‘neither’
    endpoints. If bounds are timestamps defaults to both. If bounds are
    integer locations then defaults to left.

    """
    bound = _set_window_bounds(series, bounds, closed)
    return series[bound[0]:bound[1]]


# def _create_vel_header():
#     pass


def _subset_from_windows(df: FrameOrSeries, windows) -> pd.DataFrame:
    df = df.to_frame() if type(df) == pd.Series else df
    subset = pd.DataFrame()
    for w in windows:
        data_in_subset = df[w[0]] if len(w) == 1 else df[w[0]:w[1]]
        if not data_in_subset.empty:
            subset = subset.append(data_in_subset)
    return subset


def determine_stn_flow_dist(stnobj1, stnobj2, comp_date):
    '''
    Inputs:
        stnobj1   - GPS object
        stnobj2   - GPS object
        comp_date - tuple of date strings

    Output:


    '''
    n0 = stnobj1.dnorth[comp_date[0]].mean()
    e0 = stnobj1.deast[comp_date[0]].mean()
    n1 = stnobj2.dnorth[comp_date[1]].mean()
    e1 = stnobj2.deast[comp_date[1]].mean()
    dn = n1-n0
    de = e1-e0
    dist = sqrt(dn**2 + de**2)
#     print(f'{dn}, {de}')
#     print(f'Distance traveled by {stnobj1.stn}\n'
#     f'from {comp_date[0]} to {comp_date[1]} = {dist:.4f} m' )
    return dist


def print_stn_stats(stnD1, stnD2, stnV, daterange, *args):
    Vmean = stnV.X_vel.mean()
    Dist = determine_stn_flow_dist(stnD1, stnD2, daterange)
    name = stnD1.stn
    print(f'------------------------{name}---------\n'
          f'mean along flow velocity:   {Vmean:.4f} m/d\n'
          f'                           {Vmean*365:.4f} m/a\n'
          f'flow {daterange[0][5:]} of '
          f'{daterange[0][0:4]}-{daterange[1][2:4]}:'
          f'     {Dist:.4f} m/a'
          )
    for days in args:
        dist = determine_stn_flow_dist(stnD1, stnD2, days)
        print(
            f'flow {days[0][5:]} of {days[0][0:4]}-{days[1][2:4]}:     {dist:.4f} m/a')
    return


def normalize_gps_data(data1, data2, norm_val):
    Xnorm1 = data1.X_vel/norm_val
    Xnorm2 = data2.X_vel/norm_val
    return Xnorm1, Xnorm2


def infer_sampling(df):
    """inferred sampling rate in seconds
    
    Args:
        df (pd.DataFrame) : time-indexed data frame
        
    Returns:
        sampling_rate (int) : sampling rate in seconds.
    """
    num_samples = 1000 if len(df) > 1000 else randint(4, len(df)-2)
    start, end = random_index_for_slice(df, num_samples)
    lst = (df.index[start+1:end+1]-df.index[start:end]).seconds.to_list()
    return max(set(lst), key=lst.count)


def random_index_for_slice(df, length):
    slice_in_range = len(df)-2
    if length >= slice_in_range or length < 2:
        raise ValueError(
            f'length {length} not valid for input data\n'
            'length must be: 2 < length < (len(df)-2)')
    start = randint(2, slice_in_range - length)
    return start, start+length


def clip_to_window(df, window, col_name):
    """return dataframe with only wanted data

    Args:
        df (pd.DataFrame): time-indexed dataframe
        window (tuple): [description]
            str, tuple, window, pd.Period
        col_name (str): column for setting window

    Returns:
        df (pd.DataFrame): dataframe with col
    """
    if window is not None:
        window = to_exact_indexing(window, df[col_name])
        df = df[window[0]:window[1]]
    return df


def find_label(component):
    """return second tuple entry in _directions matching each letter in arg

    Parameters
    ----------
    component : str
        string with letters matching tuple entries in _directions

    Returns
    -------
    labels : list
        list of strings taken from _directions
        
        
    !NOTE: not written to have any exceptions! i.e., error will be raised
    if component has a letter not in _directions. 
    """
    labels = []
    for letter in component:
        labels.extend([name for i, name in _directions if i == letter])
    return labels


def pct_day(duration):
    """the percentage of a day for a given duration"""
    return pd.Timedelta(duration).seconds / SECONDS_PER_DAY


def either_empty(df0, df1):
    return True if df0.empty or df1.empty else False


def has_data(num_obs, *args):
    """True if all *args have length>num_obs"""
    lengths = []
    for arg in args:
        lengths.append(len(arg))
    return True if min(lengths) > num_obs else False


def is_good(df0, df1, num_obs):
    """True if df's aren't empty and have more than 3 observations"""
    is_good = False
    if not either_empty(df0, df1):
        is_good = has_data(num_obs, df0, df1)
    return is_good


def get_range(idx, shift):
    return (idx - shift, idx + shift)


def adjust_end(df, start, end):
    return df[start:end].index[-2] if end in df[start:end].index else end


def adjust_start(df, start, end):
    return df[start:end].index[1] if start in df[start:end].index else start


def _set_window_bounds(data, bounds, closed):
    check_input(closed, None, 'left', 'right', 'both', 'neither')
    start, end = bounds
    if len(data[start:end]) > 3:
        if closed in ['right', 'neither']:
            start = adjust_start(data, start, end)
        if closed in ['left', 'neither']:
            end = adjust_end(data, start, end)
    return start, end


def check_input(value, *args):
    if value not in args:
        raise ValueError(
            f'input value ({value}) is not a valid option\n'
            f'valid options are: {args}')
    pass


def position_subsets(df, idx, t_shift, stat_shift, closed=None):
    """
    

    Parameters
    ----------
    df : _type_
        _description_
    idx : _type_
        _description_
    t_shift : str
        delta t or timespan between observations subsets
        separation window/2
    stat_shift : _type_
        length of time to preform statistics on to get values at either
        side of t_shift for calcualtion. stat window/2
    closed : _type_, optional
        _description_, by default left

    Returns
    -------
    _type_
        _description_
    """
    t0, t1 = get_range(idx, t_shift)
    x_bounds = tuple(map(lambda x: get_range(x, stat_shift), (t0, t1)))
    return tuple(map(lambda x: subset_from_bounds(df, x, closed), x_bounds))


def vel_equ(df0: pd.DataFrame, df1: pd.DataFrame, col_name: str, dt: float):
    """calculate velocity in meters per day

    Args:
        df0 (pd.DataFrame): time-indexed positions for x0 (unit: m)
        df1 (pd.DataFrame): time-indexed positions for x1 (unit: m)
        col_name (str): column name for direction in df0 and df1
        dt (float): time between positions in fraction of a day

    Returns:
        vel_mpd (float): velocity mpd in the col_name assocaited direction
    """
    return (df1[col_name].median() - df0[col_name].median()) / dt

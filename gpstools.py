#!/user/bin/env python

from math import atan, sin, cos, sqrt
from datetime import timedelta
from pathlib import PurePath, Path
import time
from typing import List, Tuple, Optional, Union, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from datatools.constants import SECONDS_PER_DAY
from datatools.utils import c_rolling
# <TODO: automatically see which columns are in file and import accordingly>

FrameOrSeries = Union[pd.DataFrame, pd.Series]
WindowTypes = Union[Tuple[str], Tuple[str, str]]
WindowList = List[WindowTypes]


# Dictionaries
station_names = (
    ('USF1', 'LMID'),
    ('USF2', 'JEME'),
    ('USFL', 'JEME'),
    ('USF3', 'JNIH'),
    ('USF4', 'EORM'),
    ('USF5', 'CMID'),
    ('USF6', 'HMID'),
    ('USF7', 'RADI'),
    ('USF8', 'MARS'),
    ('USFN', 'MARS'),
    ('USFK', 'ROCK')
)

EGM2008 = {
    'lmid': 29.0059,
    'jeme': 28.9984,
    'jnih': 28.9736,
    'cmid': 29.3225,
    'sbpi': 29.3352,
    'eorm': 29.4811,
    'hmid': 29.2759,
    'radi': 29.4067,
    'mars': 29.2258,
    'kaga': 28.3183,
    'rock': 27.1274
}
EGM96 = {
    'lmid': 29.2909,
    'jeme': 29.2815,
    'jnih': 29.2555,
    'cmid': 29.6720,
    'sbpi': 29.6949,
    'eorm': 29.8696,
    'hmid': 29.6407,
    'radi': 29.7915,
    'mars': 29.5963,
    'kaga': 29.0784,
    'rock': 27.6441
}


# background velocities for stations determined
# using constant background motion
# lmid ['2018-05-22':'2018-05-26']
# jeme ['2018-5-21':'2018-05-26']
# jnin ['2018-05-21':'2018-5-26']
background_velocities_mpd = (
    ('LMID', 0.244),
    ('JEME', 0.240),
    ('JNIH', 0.258),
    ('RADI', 0.118)
)


_directions = (
    ('n', 'dnorth'),
    ('e', 'deast'),
    ('u', 'dheight'),
    ('x', 'xflow'),
    ('t', 'xtran'),
)

# Station Info -- TEMP
# Rock Station Po


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
        self.geoid_height = [value
                             for key, value in EGM2008.items()
                             if key.upper() == self.name][0]
        self.elevation = self.ellipsoidal_height - self.geoid_height

    def __str__(self):
        return 'Base Station Object'

    def __repr__(self):
        return 'Base Station '+self.name


rock = BaseStn('ROCK',
               site_pos=(1412215.2584, -1711212.5767, 5960386.7316),
               geod_pos=(69.708219352, 309.531891746, 594.5942))

kaga = BaseStn('KAGA',
               site_pos=(1464296.0967, -1733658.2881, 5940955.0164),
               geod_pos=(69.222301946, 310.185368004, 150.0098))

defined_base_stations = {
    'ROCK': rock,
    'KAGA': kaga
}

# dictionary of antenna adjustments
# adjustment (positive is raising the antenna, unit meters)
antenna_adjustments = {
    'LMID': {
        'date': pd.Timestamp('2018-07-07 13:28:00'),
        'adjustment': -2,
        'drop_until': pd.Timestamp('2018-07-07 21:40:00')
    },
    'JEME': {
        'date': pd.Timestamp('2018-07-17 23:40:00'),
        'adjustment': -2,
        'drop_until': pd.Timestamp('2018-07-18 00:45:00')
    }
}


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

    stn_name = [name for i, name in station_names if i == stn_ID]
    if not stn_name:
        stn_name = [name for i, name in station_names if name == stn_ID]
    return stn_ID, stn_name[0]


# class Station:
#     def __init__(self, file, **kwargs):
#         self.info = None
#         self.name = None
#         self.ID = None


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
    ---------------
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
        self.base_stn = self._infer_base_stn(base_stn)
        try:
            if self.stn == 'LMID':
                self.data = self.data.drop(
                    index=self.data['2018-3-23 01:30':'2018-3-25'].index)
        except:
            pass
        # * Change assigning attrs from explicit to implicit using a dic maybe
        # * with a list of col names

        self.dnorth = self.data.dnorth
        self.deast = self.data.deast

        self.z_without_lowering_correction = self.data['dheight']
        self.z = self._antenna_lowering_correction([2018])
        self.ellipsoidal_height = self._calc_ellipsoidal_height()
        self.elevation = self._calc_elevation(
        ) if self.ellipsoidal_height is not None else None

        if 'dnorth_err' in self.data:
            self.errs = pd.DataFrame(
                {'N': self.data.dnorth_err, 'E': self.data.deast_err,
                    'U': self.data.dheight_err})

        if _is_file(gps_data):
            # direction_diff is shifted to the origin (data - data at t=0)
            # this means each timeseries will start at 0 (origin)
            self.dnorth_diff = self.dnorth - self.dnorth.dropna()[0]
            self.deast_diff = self.deast - self.deast.dropna()[0]

            self.horizontal_disp = np.sqrt(self.deast_diff**2
                                           + self.deast_diff**2)

            self.dnorth_daily = self.dnorth.resample('1D').mean().dropna()
            self.deast_daily = self.deast.resample('1D').mean().dropna()
            # NOTE np.atan
            self.alpha = atan((self.dnorth_daily[-1] - self.dnorth_daily[0])
                              / (self.deast_daily[-1] - self.deast_daily[0]))

            self.xflow = -1 * (cos(self.alpha) * self.deast_diff
                               + sin(self.alpha) * self.dnorth_diff)

            self.xtran = (-1 * sin(self.alpha) * (self.deast_diff)
                          + cos(self.alpha) * (self.dnorth_diff))

            self.data['xtran'] = self.xtran
            self.data['xflow'] = self.xflow
        else:
            self.xflow = self.data['xflow']
            self.xtran = self.data['xtran']

        self.dist_from_basestn = np.sqrt(
            self.data.dnorth ** 2 + self.data.deast ** 2)
        self.data['Dist'] = self.dist_from_basestn

        self.baseline = np.sqrt(self.dnorth[0]**2+self.deast[0]**2)
        self.sampling_rate = infer_sampling(self.data)
        self.vel_header = None

        # if 'usf118' in self.file:
        #     t1 = pd.to_datetime('2018-07-07 16:28:29')
        #     t2 = pd.to_datetime('2018-07-07 17:06:00')
        #     z_adj = self.data.dheight[t1]-self.data.dheight[t2]
        #     self.z_corr = self.z
        #     self.z_corr[t2:] = self.z_corr[t2:]+z_adj

    def __str__(self):
        return print('gps data')

    def _infer_base_stn(self, base_stn_kw):
        base_stn_kw = 'KAGA' if self.year == 2017 else base_stn_kw
        name_in = self.file_name.name if base_stn_kw is None else base_stn_kw
        base_stn = [obj for name, obj in defined_base_stations.items()
                    if name.upper() in name_in]
        return base_stn[0] if len(base_stn) > 0 else None

    def _antenna_lowering_correction(self, correct_years: list) -> pd.Series:
        """correct vertical gps position for antenna adjustments.




        """
        if self.stn in antenna_adjustments and self.year in correct_years:
            info = antenna_adjustments[self.stn]
            adjust_at = info['date']

            z_adj = self.data['dheight'][adjust_at:]-info['adjustment']
            # modify dataframe self.data with adjusted height
            self.data.loc[z_adj.index, 'dheight'] = z_adj

            if info['drop_until'] is not None:
                self.data = self.data.drop(
                    self.data[adjust_at:info['drop_until']].index)
        return self.data[['dheight']]

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

    def drop_positions(self, droplist: list, apply_to_instance=False):
        dropDF = _subset_from_windows(self.data, droplist)
        if apply_to_instance:
            self.drop_from_instance(dropDF.index)
        return dropDF

    def _calc_ellipsoidal_height(self) -> Union[FrameOrSeries, None]:
        """base-onIce height difference to ellipsoidal height in me"""
        ellip_hgt = None
        if self.base_stn is not None:
            ellip_hgt = self.base_stn.ellipsoidal_height + self.data.dheight
            ellip_hgt = ellip_hgt.rename('ellipsoidal_height')
        return ellip_hgt

    def _calc_elevation(self):
        local_geoid_height = [val for stn, val in EGM2008.items()
                              if stn.upper() == self.stn][0]
        elev = self.ellipsoidal_height - local_geoid_height
        return elev.rename('elevation')

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
        df = clip_to_window(data, window, col_name=['doy', component])
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
        from scipy import signal
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
                      #   smoothing: Optional[str] = None,
                      #   set_min_periods: Optional[Union[int, None]] = False,
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
            smoothing_window (int, optional):
                Rolling mean window length applied to positions before
                calculating velocity. No smoothing applied if None.
                Defaults to None.
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

        # if smoothing is not None:
        #     min_periods = self.samples_in_timespan(
        #         smoothing) if set_min_periods is False else set_min_periods
        #     df = c_rolling(df, smoothing, min_periods=min_periods)

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

    def _create_vel_header(component,
                           stat_window,
                           separation_window,
                           window):
        header = ''
    pass

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

    def calc_NEUXTHvel(self,
                       calc_only='x',
                       seperation_window=2,
                       averaging_period=3,
                       robust=None):
        """
        adapted from Surui's script, default naming remanes

        only_calc_XT = None or 0 (calculate NEUXT)
                                = 1 (True) only calculate XT
        Function to calculate velocity in the NEU and XT directions
        returns velocity in m/d
        calculates vel every 3 minutes using two 3-min periods seperated
                by every 2 hours
        default values:
        seperation_widow = 2 (hours)
        averaging_period = 3 (min)

        Notes
        mt - % of the day of 3 minutes
        dt - % of the day for seperation window
        vt - array of times(gps)

        """
        print("method is deprecated, use method: calc_velocity")
        # if seperation_window == None:
        #     seperation_window = 2
        # if averaging_period == None:
        #     averaging_period = 3
        # sec_in_period = int((averaging_period * 60) / 2)
        # min_in_window = seperation_window * 60

        # mt = np.timedelta64(sec_in_period, 's')
        # dt = np.timedelta64(seperation_window, 'h')

        # # fraction of a day
        # dt2 = 0.000347 * 2.0 * min_in_window
        # # if robust == None:
        # timevec = np.arange(self.date[0] + dt,
        #                     self.date[-1] - dt
        #                     + timedelta(seconds=0.1), mt*2)
        # # else:
        # # timevec = self.date
        # tVNEU0 = np.empty(0)
        # import time
        # t = time.time()
        # for j in range(0, len(timevec)):
        #     tj0 = timevec[j] - dt
        #     tj1 = timevec[j] + dt
        #     sID00 = np.where(((self.date - tj0) >= (-1 * mt))
        #                      & ((self.date - tj0) < mt))[0]
        #     sID01 = np.where(((self.date - tj1) >= (-1 * mt))
        #                      & ((self.date - tj1) < mt))[0]
        #     if ((len(sID00) > 3) & (len(sID01) > 3)):
        #         if calc_only == 'x':
        #             VXj = (np.median(self.xflow[sID01])
        #                    - np.median(self.xflow[sID00])) / (2.0 * dt2)
        #             tVNEU0 = np.append(tVNEU0, [timevec[j], VXj])
        #             num_cols = int(2)
        #             column_names = ['Date', 'X_vel']
        #         else:
        #             VNj = (np.median(self.dnorth[sID01])
        #                    - np.median(self.dnorth[sID00])) / (2.0 * dt2)
        #             VEj = (np.median(self.deast[sID01])
        #                    - np.median(self.deast[sID00])) / (2.0 * dt2)
        #             VUj = (np.median(self.z[sID01])
        #                    - np.median(self.z[sID00])) / (2.0 * dt2)
        #             VXj = (np.median(self.xflow[sID01])
        #                    - np.median(self.xflow[sID00])) / (2.0 * dt2)
        #             VTj = (np.median(self.xtran[sID01])
        #                    - np.median(self.xtran[sID00])) / (2.0 * dt2)
        #             VHj = (np.median(self.dist_from_basestn[sID01])
        #                    - np.median(self.dist_from_basestn[sID00])) / (2.0 * dt2)

        #             tVNEU0 = np.append(tVNEU0,
        #                                [timevec[j], VNj, VEj, VUj, VXj, VTj, VHj])
        #             num_cols = int(7)
        #             column_names = ['Date', 'N_vel', 'E_vel', 'U_vel',
        #                             'X_vel', 'T_vel', 'H_vel']
        #     else:
        #         continue
        # a = int(len(tVNEU0) / num_cols)
        # tVNEU = tVNEU0.reshape(a, num_cols)
        # vel_data = pd.DataFrame(tVNEU, columns=column_names)
        # vel_data = vel_data.set_index(['Date'])
        # elapsed = time.time() - t
        # print(f'Time Elapsed: {elapsed:.1f} seconds')
        # return vel_data
        pass


# FUNCTIONS

def _is_file(obj):
    return isinstance(obj, (Path, PurePath, str))


def _get_data(gps_data: Union[Path, PurePath, str, FrameOrSeries]):
    return gps_data if isinstance(gps_data, pd.DataFrame) else load_NEUgps(gps_data)


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


def loadAndrewsNEU(file):
    col_names = ['station', 'doy', 'deast', 'dnorth', 'dheight',
                 'deast_err', 'dnorth_err', 'dheight_err', 'rms',
                 'DD', 'atm', 'datm', 'rho_ua', 'numbBF', 'NFbias']
    df = pd.read_csv(file, index_col=1, parse_dates=True, names=col_names,
                     dtype=np.float64)
    df.drop(columns='station', inplace=True)
    return df


def loadAndrewsVel(file):
    col_names = ['doy', 'xvel']
    df = pd.read_csv(file, index_col=0, parse_dates=True, names=col_names,
                     dtype=np.float64)
    return df


def loadNEUwerrs(file):
    col_names = ['doy', 'dnorth', 'dnorth_err', 'deast',
                 'deast_err', 'dheight', 'dheight_err', 'rms',
                 'dd', 'bf', 'err_code']
    df = pd.read_csv(file, index_col=0, parse_dates=True, names=col_names,
                     dtype={'index': '<M8[ns]', 'doy': np.float64,
                            'dnorth_err': np.float64, 'deast': np.float64,
                            'deast_err': np.float64, 'dheight': np.float64,
                            'dheight_err': np.float64, 'rms': np.float64,
                            'dd': np.int64, 'bf': np.int64, 'err_code': np.int64},
                     na_values='  nan')
    df.drop(columns='err_code', inplace=True)
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


def _create_vel_header():
    pass


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
    """inferred sampling rate in seconds"""
    start, end = random_index_for_slice(df, 1000)
    lst = (df.index[start+1:end+1]-df.index[start:end]).seconds.to_list()
    return max(set(lst), key=lst.count)


def random_index_for_slice(df, length):
    slice_in_range = len(df)-2
    if length >= slice_in_range or length < 2:
        raise ValueError(
            f'length {length} not valid for input data\n'
            'length must be: 2 < length < (len(df)-2)')
    from random import randint
    start = randint(2, slice_in_range - length)
    return start, start+length


def clip_to_window(df, window, col_name):
    """return dataframe with only wanted data

    Args:
        df (pd.DataFrame): time-indexed dataframe
        window (tuple): [description]
        col_name (str): column for setting window

    Returns:
        df (pd.DataFrame): dataframe with col
    """
    from datatools.diurnal import to_exact_indexing
    if window is not None:
        window = to_exact_indexing(window, df[col_name])
        df = df[window[0]:window[1]]
    return df


def find_label(component):
    labels = []
    for letter in component:
        labels.extend([name for i, name in _directions if i == letter])
    return labels


def pct_day(duration):
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
    "True if df's aren't empty and have more than 3 observations"
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

# Code in progress


def clean_velocity(df, file):
    if 'LMID17' in file:
        drop_list = ['2017-07-26 00:30:00', '2017-07-25 20:27',
                     '2017-07-25 20:30', '2017-07-24 07:51',
                     '2017-07-24 07:54', '2017-07-24 08:18']
        idxDrop = df[df['X_vel'] < 0.062].index
        for val in drop_list:
            idxDrop = idxDrop.append(df[df.index == val].index)
        tmp = df.X_vel['2017-07-28':'2017-08-06']
        idxDrop = idxDrop.append(tmp[0.472 < tmp].index)
        tmp = df.X_vel['2017-08-09':]
        idxDrop = idxDrop.append(tmp[0.488 < tmp].index)
        tmp = df.X_vel['2017-07-29']
        idxDrop = idxDrop.append(tmp[tmp < 0.16].index)
    elif 'MARS17' in file:
        idxDrop = df[0.35 < df.X_vel].index
        idxDrop = idxDrop.append(df[df.X_vel < 0].index)
    elif 'EORM17' in file:
        idxDrop = df[0.35 < df.X_vel].index
        idxDrop = idxDrop.append(df[df.X_vel < 0].index)
    elif 'JNIH17' in file:
        drop_list = ['2017-07-25 20:27:00', '2017-07-26 00:27',
                     '2017-07-26 05:57']
        idxDrop = df[df.X_vel < 0.047].index
        for val in drop_list:
            idxDrop = idxDrop.append(df[df.index == val].index)
    elif 'LMID18' in file:
        drop_list = ['2018-07-19 01:21:30', '2018-09-29 04:27:30',
                     '2018-08-18 05:57', '2018-08-18 18:33']
        idxDrop = df[df.X_vel < 0.140].index
        for val in drop_list:
            idxDrop = idxDrop.append(df[df.index == val].index)
        tmp = df.X_vel['2018-08-04 00:00':'2018-08-04 01:15']
        idxDrop = idxDrop.append(tmp.index)
        # tmp = df.X_vel['2018-07-27':]
        # idxDrop = idxDrop.append(tmp[0.6 < tmp].index)
        tmp = df.X_vel[:'2018-07-12']
        tmp = tmp.append(df.X_vel['2018-07-18'])
        idxDrop = idxDrop.append(tmp[0.5 < tmp].index)
        tmp = df.X_vel['2018-07-20']
        idxDrop = idxDrop.append(tmp[0.4 < tmp].index)
        # tmp = df.X_vel['2018-07-27':'2018-07-29']
        # idxDrop = idxDrop.append(tmp.index)
        # tmp = df.X_vel['2018-07-26 12:00:00':'2018-07-29']
        # idxDrop = idxDrop.append(tmp[0.32 < tmp].index)
        tmp = df.X_vel['2018-07-15':'2018-07-25']
        idxDrop = idxDrop.append(tmp[tmp < 0.207].index)
        tmp = df.X_vel['2018-08-22':]
        idxDrop = idxDrop.append(tmp[0.35 < tmp].index)
    elif 'JEME18' in file:
        drop_list = ['2018-07-24 07:50:00', '2018-06-28 22:58:30']
        idxDrop = df[df.X_vel < 0.1].index
        for val in drop_list:
            idxDrop = idxDrop.append(df[df.index == val].index)
        tmp = df.X_vel[:'2018-05-20']
        tmp = tmp.append(df.X_vel['2018-07-18':'2018-07-20'])
        idxDrop = idxDrop.append(tmp.index)
        tmp = df.X_vel[:'2018-06-18']
        idxDrop = idxDrop.append(tmp[0.47 < tmp].index)
        tmp = df.X_vel['2018-06-22':'2018-06-25']
        idxDrop = idxDrop.append(tmp[0.5 < tmp].index)
    elif 'CMID18' in file:
        idxDrop = df[df.X_vel <= 0].index
        idxDrop = idxDrop.append(df[0.38 < df.X_vel].index)
    elif 'RADI18' in file:
        idxDrop = df[df.X_vel <= 0].index
        idxDrop = idxDrop.append(df[0.3 < df.X_vel].index)
        idxDrop = idxDrop.append(df['2018-05-31'].index)
        idxDrop = idxDrop.append(df['2018-06-5'].index)
        idxDrop = idxDrop.append(df['2018-05-6'].index)
    elif 'JNIH18' in file:
        idxDrop = df.X_vel['2018-7-12'].index
        tmp = df.X_vel['2018-4-23 7:00':'2018-4-23 7:30']
        tmp = tmp.append(df.X_vel['2018-5-27 5:30':'2018-5-27 12:00'])
        tmp = tmp.append(df.X_vel['2018-6-10 21:00':'2018-6-10 23:30'])
        tmp = tmp.append(df.X_vel['2018-6-29 12:00':'2018-6-29 17:00'])
        tmp = tmp.append(df.X_vel['2018-3-27 21:00':'2018-3-27 23:10'])
        tmp = tmp.append(df.X_vel['2018-3-11 4:30':'2018-3-11 5:00'])
        idxDrop = idxDrop.append(tmp.index)
        tmp2 = df.X_vel['2018-3-18':'2018-3-27']
        idxDrop = idxDrop.append(tmp2[tmp2 < 0.175].index)
        idxDrop = idxDrop.append(df[df.X_vel <= 0].index)
        idxDrop = idxDrop.append(df[df.X_vel > 0.7].index)
    else:
        idxDrop = filter_zero(df)
    df.drop(idxDrop, inplace=True)
    return df


class station:
    """
    data and information for a single gps station

    attributes:
        station.name
        station.id
        station.
        station.environment
            rock - mounted somewhere on the earth's crust
            ice - mounted on moving ice
            roving - not mounted
        station.adjustment
        station.date_removed
    """

    def __init__(self):
        self.temp = 'temp'


class velocity:
    """



    """

    def __init__(self, file: str):
        # load velocity data into pd.Dataframe
        self.df = load_NEUXTvel(file)
        # define velocity components
        self.along_flow = self.df.X_vel
        self.transverse_flow = self.df.T_vel
        self.northing = self.df.N_vel
        self.easting = self.df.E_vel
        self.vertical = self.df.U_vel
        self.stn_ID, self.stn = get_station_name(file)

    def __str__(self):
        return print(f'NEUXT calculated velocities for station')

    def rolling_smooth(self, window=6, num_periods=None):
        """
        smooth velocity data with a rolling window average filter

        Parameters
        =========

        Returns
        =======

        Usage
        =====
        from gpstools import velocity
        stn1 = velocity('./'STN118_VEL_NEUXT.csv')
        figure()
        plot(stn1.x.rolling_smooth())
        plot(stn1.x.rolling_smooth(window=3,num_periods=24),'.')
        title('Compare 6 & 3 hr smoothing for stn1 along flow ice vel')

        """
        # win = str(window)+'H'
        # t_shift = window/2
        # if num_periods:
        #     num = num_periods
        # else:
        #     num = window*4

        pass


# if __name__ == "__main__":
#     import sys
#     fib(int(sys.argv[1]))

"""
Suite of calculations for diurnally varying timeseries data

- jzmejia
"""


from collections import OrderedDict
import datetime
import functools
import time
import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Tuple, Optional
import matplotlib.dates as dates
import matplotlib.units as munits

# Useful decorator functions


def lag_df(midx, extrema='maxima', lag_from='ablation'):
    df = midx.xs((extrema, 'hrs'), axis=1).unstack()
    return df.apply(lambda x: x - df[lag_from]).drop(columns=[lag_from])


def dhrs_to_timedelta(dhrs):
    """convert decimal hours to pandas timestamp

    Args:
        dhrs (float): decimal hours

    Returns:
        [pd.Timedelta]: num hours
    """
    return pd.Timedelta(hours=dhrs)


def dhrs_to_timestamp(data):
    """convert a series of decimal hours to timestamps (req. date index)

    Args:
        data (series): series with a DateTimeIndex and float of decimal
        hours

    Returns:
        [series]: same index as data but values as timestamps
    """
    return data.index + data.apply(dhrs_to_timedelta)


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


def set_unit(unit):
    """Register a unit on a function
    @set_unit("cm^3")
    def volume(radius, height):
        return math.pi * radius**2 * height
    """
    def decorator_set_unit(func):
        func.unit = unit
        return func
    return decorator_set_unit


def plot_extrema(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #
        df = func(*args, **kwargs)
        if df is not None:
            plt.figure()
            timeseries = args[0].timeseries
            plt.plot(timeseries, linewidth=1)
            # plt.plot(timeseries,'.c',markersize=1)
            plt.plot(df.min_time, df.min_val, '.')
            plt.plot(df.max_time, df.max_val, '.r')
            plt.ylabel(timeseries.name)
        else:
            df = DiurnalExtrema.find_diurnal_extrema
        return df
    return wrapper

# DECORATOR WARNING - NOT WORKING ITS MAKING THE DF RETURN NONE
# def threshold(func):
#     @functools.wraps(func)
#     def apply_threshold(*args, **kwargs) -> pd.DataFrame:
#         df = func(*args, **kwargs)
#         df = df.dropna(how='any')
#         threshold = args[0].threshold
#         # check threshold is valid
#         if threshold is not None:
#             if 0 < threshold <= 1:
#                 min_amplitude = (df.eval('max_val - min_val')).mean()*threshold
#                 for day, vals in df.iterrows():
#                     daily_amp = vals.max_val - vals.min_val
#                     if daily_amp <= min_amplitude:
#                         df = df.drop(day)
#             else:
#                 raise ValueError(
#                     'threshold entred is not a valid decimal percent')
#             return df
#     return apply_threshold


# Format date ticks using ConciseDateFormatter

converter = dates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter


# class ExtremaMixIns(object):
#     def __init__(self):
#         pass


# class Extrema(object):
#     def __init__(self, timeseries, extrema_picks, stn=None):
#         self.timeseries = timeseries
#         self.df = readin(extrema_picks)
#         self.stn = stn

#     def readin(self, extrema_picks):
#         if isinstance(extrema_picks, (str, Path)):
#             df = pd.read_csv(extrema_picks, index_col=0, parse_dates=True)
#         elif isinstance(extrema_picks, pd.DataFrame):
#             df = extrema_picks
#         else:
#             raise ValueError(
#                 f'extrema_pick type {type(extrema_picks)} unrecognized.'
#             )
#         return df

#     def extrema_index(self, which="max"):
#         """ Ex: self.extrema_series("min")
#             Default: self.extrema_series()
#         """
#         return pd.DataFrame(data={which: self.df[which+"_val"].values},
#                             index=self.df[which+"_time"].values)

#     def amplitude(self):
#         amp = self.df['max_val']-self.df['min_val']
#         amp.index = amp.index.to_timestamp()
#         return amp


class DiurnalExtrema(object):
    """
    Timeseries with diurnally varying vals(1 max and 1 min per 24-hrs)

    Arguments:
        timeseries {pd.Series[DatetimeIndex, float64]}: Timeseries data.

        min_first [bool]
            Defaults to True.
        max_next_day[bool]
            Defaults to True.
        predict_timing [bool]
            Defaults to False.
        window [Tuple, Int] : timeperiod to use when predicting timing.
            Defaults to 4.
        threshold --not working yet (deletes extrema picks with a diurnal
            amplitude below )
            Defaults to None.
        generate_figure
            Defaults to False.
        stn
            Defaults to None.

    """

    def __init__(self,
                 timeseries,
                 min_first=True,
                 max_next_day=True,
                 predict_timing=False,
                 window=4,
                 threshold=None,
                 generate_figure=False,
                 stn=None,
                 **kwargs):

        self.timeseries = timeseries.dropna()
        if self.timeseries.index.tz is not None:
            self.timeseries = self.timeseries.tz_localize(None)
        self.threshold = threshold
        self.min_first = min_first
        self.max_next_day = max_next_day
        self.predict_timing = predict_timing
        self.window = window
        # self.threshold = threshold
        self.stn = stn
        self.generate_figure = generate_figure
        self.diurnal_vals = []
        self.df = self.find_diurnal_extrema()
        # self.extrema = Extrema(self.timeseries, self.df)
        if self.generate_figure:
            self.plot()

    # def __add__(self, other):
    #     """merge extrema picks if station is the same"""
    #     if self.stn != other.stn:
    #         raise ValueError(
    #             f'stations are not compatable, to override rename'
    #         )
    #     return

    def decimal_hours(self, which="max", check_before=12):
        """timestamp indexed extrema time pick in decimal hours (0-24)."""
        decimal_hrs = (self.df[which+'_time'] - self.df.index
                       ).dt.total_seconds()/(60*60)
        decimal_hrs = decimal_hrs.rename("hrs")
        # check and correct for next-day extrema picks

        return decimal_hrs

    # def check_decimal_hour_calc(self, decimal_hours, which, check_before):
    #     early_picks = decimal_hours[decimal_hours < check_before]
    #     if len(early_picks) != 0:
    #         for idx, hrs in early_picks.iteritems():
    #             true_time = self.df.loc[idx.to_period('D'), which+'_time']
    #             time_diff = true_time-idx
    #             hrs_new = (time_diff.days * 24) + (time_diff.seconds/(60*60))
    #             if hrs_new != hrs:
    #                 decimal_hours.loc[idx] = hrs_new

    #     return decimal_hours

    def extrema_index(self, which="max"):
        """ Ex: self.extrema_series("min")
            Default: self.extrema_series()
        """
        return pd.DataFrame(data={which: self.df[which+"_val"].values},
                            index=self.df[which+"_time"].values)

    def truncated_picks(self, date_range, value, which):
        """identifies picks on a truncated timeseries and creates attr.

        Args:
            idx (list of str): dates where extrema pick is truncated.
            which (str, optional): extrema that is truncated.
                Defaults to 'min'
        """

        pass

    def amplitude(self):
        amp = self.df['max_val']-self.df['min_val']
        amp.index = amp.index.to_timestamp()
        return amp

    def find_diurnal_extrema(self):
        if self.predict_timing:
            minOccurs, maxOccurs = self.predict_extrema()
            # print(
            #     f'predicted extrema timing:\n'
            #     f'    minimum: {minOccurs}\n'
            #     f'    maximum: {maxOccurs}\n')
        for day in self.timeseries.index.to_period('D').unique():

            if self.predict_timing:
                min_window, max_window = get_occurance_windows(day, 8,
                                                               minOccurs,
                                                               maxOccurs)
                minVal, minTime = self.get_real_extrema(
                    self.timeseries, min_window, 'min')
                if self.min_first and minTime and max_window[0] < minTime:
                    max_window = (minTime, max_window[1])
                maxVal, maxTime = self.get_real_extrema(
                    self.timeseries, max_window, 'max')

            else:
                # find diurnal minimum
                minVal, minTime = self.get_real_extrema(
                    self.timeseries, day, 'min')
                if self.min_first and self.max_next_day:
                    # find diurnal maximum in 18 hour window after minimum
                    maxVal, maxTime = self.get_real_extrema(self.timeseries,
                                                            (minTime, minTime
                                                             + pd.Timedelta(hours=18)),
                                                            'max')
                elif not self.min_first or not self.max_next_day:
                    maxVal, maxTime = self.get_real_extrema(
                        self.timeseries, day, 'max')

            # check max value is larger than min
            if maxVal and minVal and minVal > maxVal:
                continue
            self.diurnal_vals.append({'Date': day.to_timestamp(),
                                      'min_val': minVal,
                                      'min_time': minTime,
                                      'max_val': maxVal,
                                      'max_time': maxTime})

        self.df = pd.DataFrame(self.diurnal_vals).set_index('Date')
        self.apply_threshold()
        return self.df

    def multi_indexed(self):
        self.df['min_hrs'] = (self.df.min_time
                              - self.df.index).dt.total_seconds()/(60*60)
        self.df['max_hrs'] = (self.df.max_time
                              - self.df.index).dt.total_seconds()/(60*60)
        df = self.df[['min_val', 'min_time', 'min_hrs',
                      'max_val', 'max_time', 'max_hrs']]

        col_labels = [np.array(['minima', 'minima', 'minima',
                                'maxima', 'maxima', 'maxima']),
                      np.array(['value', 'time', 'hrs',
                                'value', 'time', 'hrs'])]
        self.multi = pd.DataFrame(
            df.values, index=self.df.index, columns=col_labels)
        self.multi = self.multi.astype(dtype={('minima', 'value'): float,
                                              ('minima', 'hrs'): float,
                                              ('maxima', 'value'): float,
                                              ('maxima', 'hrs'): float
                                              })
        return self.multi

    def apply_threshold(self):
        if self.threshold is not None:
            self.df.drop(self.df[(self.df.max_val - self.df.min_val)
                                 < self.threshold].index, inplace=True)
        pass

    def change_extrema_picks(self,
                             day,
                             which: str,
                             new_extrema_value=None,
                             new_extrema_time=None,
                             find_between=False,
                             find_near=False
                             ):
        """Change extrema picked by find_diurnal_extrema.x
        Args:
            day (Union[str, pd.Period, pd.DatetimeIndex]): extrema index
            new_extrema (tuple or str): (extrema value, extrema time)
                or none
            which (str): which extrema to change
                options = 'min', 'max', 'both'
            find_between (tuple, floats or ints): find extrema value 
                between first and last entry of tuple (format, hours after
                index)
        """

        if hasattr(self, 'diurnal_extrema_picks') is False:
            self.diurnal_extrema_picks = self.df

        check_input(which, 'min', 'max', 'both')
        idx = pd.Timestamp(day)

        # check if index is in df already if not you are adding not changing
        if isinstance(self.df.index, pd.DatetimeIndex) and idx not in self.df.index:
            raise ValueError(f'date={day} not found in dataframe index')

        extrema = ['min', 'max'] if which == 'both' else [which]

        if find_between:
            t0, t1 = add_hours(day, find_between)
            value, time = self.get_extrema(self.timeseries[t0:t1], which)
            self.update_extrema(which, idx, value, time)
        elif not new_extrema_value and not new_extrema_time:
            for which in extrema:
                self.update_extrema(which, idx, None, None)
        else:
            if new_extrema_value:
                self.update_extrema_comp(
                    which+'_val', idx, new_extrema_value)
            if new_extrema_time:
                self.update_extrema_comp(
                    which+'_time', idx, pd.Timestamp(new_extrema_time))
                if not new_extrema_value and find_near:
                    self.update_extrema_comp(
                        which+'_val', idx,
                        self.value_around_time(
                            new_extrema_time, return_max=(which == 'max')))

        pass

    def value_around_time(self, time, dt=15, return_max=True):
        """

        Args:
            time (str): time to search around in timeseries.
            dt (int, float) : number of minutes to search about time.
            choose_by (str) : how to choose value. 
                Currently available options are 'max' or 'min'
                Defaults to 'max'
        """

        subset = self.timeseries[pd.Timestamp(time)-pd.Timedelta(minutes=dt):
                                 pd.Timestamp(time)+pd.Timedelta(minutes=dt)]
        return subset.max() if return_max else subset.min()

    def update_extrema_comp(self, column, idx, new_value):
        self.df.loc[idx, column] = new_value
        pass

    def update_extrema(self, extrema, idx, value, time):
        self.update_extrema_comp(extrema+"_val", idx, value)
        self.update_extrema_comp(extrema+"_time", idx, time)
        pass

    @functools.lru_cache()
    def predict_extrema(self):
        """Return average time of extrema occurance"""

        # temp (already in main script)###### CHECK THIS PART
        if self.timeseries.index.tz is not None:
            # unused variable
            timezone_info = str(self.timeseries.index.tzinfo)
            self.timeseries = self.timeseries.tz_localize(None)
        #########    ##########

        start_time, end_time = to_exact_indexing(self.window, self.timeseries)
        calib_data = self.timeseries[start_time:end_time]
        if calib_data.empty:
            raise ValueError('Calibration timerange is not valid')

        # initialize for loop
        max_occurs, min_occurs = [], []
        for day in calib_data.index.to_period('D').unique():

            # Find extrema for day in window
            minima = self.get_real_extrema(calib_data, day, 'min')
            min_occurs = add_occurance(
                minima[1]-day.to_timestamp(), min_occurs)

            if minima[1] and self.min_first and self.max_next_day:
                # find diurnal maximum in 18 hour window after minimum
                maxima = self.get_real_extrema(calib_data,
                                               (minima[1], minima[1]
                                                + pd.Timedelta(hours=18)),
                                               'max')
            else:
                maxima = self.get_real_extrema(calib_data, day, 'max')
            max_occurs = add_occurance(
                maxima[1]-day.to_timestamp(), max_occurs)
        return mean_occurance(min_occurs), mean_occurance(max_occurs)

    def get_real_extrema(self, ts, window, min_or_max):
        '''
        returns a valid extrema (value,index_time) for given window of timeseries
        extrema pick is checked if it lies on the boundary of the window, if so
        the timeseries is extended by 5 measurements to confirm extrema is real.

        Parameters
        ----------
            ts - timeseries : series
                timeseries with length greater than window
            window : tuple, list, string, pd.Period
            min_or_max : str
                only 'min' or 'max' valid
        Returns
        -------
            extTuple : tuple
                (extremaValue, extremaIndex)
        '''
        check_input(min_or_max, 'min', 'max')
        window = to_exact_indexing(window, ts)
        extremaTuple = (None, None)
        if window is not None:
            extremaTuple = self.get_extrema(
                ts[window[0]:window[1]], min_or_max)
            if not on_boundary(ts, window, extremaTuple, min_or_max):
                extremaTuple = (None, None)
        return extremaTuple

    def get_extrema(self, ts: pd.Series, min_or_max: str) -> tuple:
        """for  ts - timeseries : series
                min_or_max : str 
        returns tuple of extrema (value, time of occurance)"""

        check_input(min_or_max, 'min', 'max')
        return get_max(ts) if min_or_max == 'max' else get_min(ts)

    def plot(self, generate_figure=False, *args, **kwargs):
        if not self.generate_figure or not generate_figure:
            self.fig = plt.figure()
            ax = self.fig.add_subplot(111)
            ax.plot(self.timeseries, linewidth=1)
            # plt.plot(timeseries,'.c',markersize=1)
            ax.plot(self.df.min_time, self.df.min_val, 'o', markersize=3)
            ax.plot(self.df.max_time, self.df.max_val, 'or', markersize=3)
            ax.set_ylabel(self.timeseries.name)
            if self.stn is not None:
                ax.set_title(self.stn)
            return ax

    def plot_extrema_picks(self, ax=None, *args, **kwargs):
        if ax is None:
            self.plot(generate_figure=True)
        else:
            ax.plot(self.timeseries)
            ax.plot(self.extrema_index(), '.', **kwargs)
            ax.plot(self.extrema_index(which="min"), '.', **kwargs)
        pass


def add_time(day, time):
    return day+' '+str(time)


def extrema_slice(day, times):
    return tuple(map(lambda x: add_time(day, x), times))


def _bool_is_same(val1, val2):
    return True if val1 == val2 else False


def hour_as_time(hrs: float) -> str:
    td = datetime.timedelta(hours=hrs)
    return f'{td.seconds//3600}:{(td.seconds//60)%60}'


def mean_occurance(occurance_list: list) -> float:
    return round(sum(occurance_list) / len(occurance_list), 3)


def get_max(ts: pd.Series) -> tuple:
    return ts.max(), ts.idxmax()


def add_hours(day, hours):
    """add number of hours in a tuple to day

    Args:
        day (str): [description]
        hours (Tuple of float or ints): number of hours to add to day

    Returns:
        [tuple(pd.Timestamp)]: timestamps specified by hours
    """
    return tuple(map(lambda hrs: pd.Timestamp(day)
                     + pd.Timedelta(hours=hrs),
                     hours))


def get_min(ts: pd.Series) -> tuple:
    return ts.min(), ts.idxmin()


def get_occurance_windows(day, window_length, *args):
    return ([create_timewindow(day, time, window_length) for time in args])


def expect_extrema_between(day, center_min, center_max, num_offset_hours):
    return ([create_timewindow(day, center, num_offset_hours) for center in [center_min, center_max]])
#     return create_timewindow(day, center_min, num_offset_hours), create_timewindow(day, center_max, num_offset_hours)


def create_timewindow(day, center, numhours):
    window_start = day.start_time + pd.Timedelta(hours=center-numhours)
    window_end = day.start_time + pd.Timedelta(hours=center+numhours)
    return window_start.round('s'), window_end.round('s')


def add_occurance(occurance_time, occurance_list):
    """
    Rounds time to nearest hour and appends to list

    Inputs: 
        occurance_time [pd.Timedelta]
        occurance_list [list]
    Returns: 
        occurance_list [list]
    """
    # print(f'occurance_time: {occurance_time}, list: {occurance_list}')
    if occurance_time is not None:
        occurance_list.append(timedelta_to_hours(occurance_time))
    return occurance_list


def check_input(value, *args):
    if value not in args:
        raise ValueError(
            f'input value ({value}) is not a valid option\n'
            'valid options are: {args}')
    pass


def check_length(data, *args):
    if len(data) not in args:
        raise ValueError(
            f'input has length={len(data)}\n'
            f'valid lengths: {args}')
    else:
        length_ok = True
    return length_ok


def start_before_end(start, end) -> bool:
    # dtypes=(pd.Timestamp, np.datetime64)
    start_first = False
    if isinstance(start, (pd.Timestamp, np.datetime64)) and isinstance(
            end, (pd.Timestamp, np.datetime64)):
        start_first = True if start < end else False
    else:
        raise TypeError(f'Argument dtypes={type(start)},{type(end)}\n'
                        f'valid dtypes are: {(pd.Timestamp, np.datetime64)}')
    return start_first


def on_boundary(timeseries,
                window: tuple,
                extrema_tuple: tuple,
                val_type: str) -> bool:
    """ 
    check if extrema is on boundary of time domain, if so, check if real

    Parameters
    ----------
    timeseries : pd.Series, datetime indexed
    window : tuple, of pd.Datetime
        time domain
    val_type : str, options 'max' or 'min'
    Returns
    -------
    boundary_ok : bool
        True - extrema not on time domain boundary or is a good extrema
        False - bad extrema (picked as artificat of time domain)
    """
    # initialize
    extrema_val, extrema_time = extrema_tuple
    start_or_end = 'start', 'end'
    boundary_ok = True
    # check the start then end of window
    for idx, bound in enumerate(window):
        # only check second bound if the first bound is ok
        if boundary_ok == True:
            # match timeseries sampling interval
            if bound not in timeseries.index:
                bound = get_index_of_bound(
                    timeseries, window, start_or_end[idx])
            if _bool_is_same(bound, extrema_time):
                boundary_ok = bool_check_around_bound(timeseries, bound,
                                                      extrema_tuple, val_type)
    return boundary_ok


def get_index_of_bound(timeseries: pd.Series, window: tuple, end_point: str):
    subset = timeseries[window[0]:window[1]]
    return subset.index[-1] if end_point == 'end' else subset.index[0]


def bool_check_around_bound(timeseries, bound, extrema_tup, val_type):
    """check around bound
    return True if pick is a good value, else false"""
    check_input(val_type, 'min', 'max')
    bound_idx = timeseries.index.get_loc(bound)
    subset = timeseries[bound_idx-5:bound_idx+5]
    if len(subset) < 3:
        is_good_val = False
    else:
        new_tup = (subset.min(), subset.idxmin())
        if val_type == 'max':
            new_tup = (subset.max(), subset.idxmax())

        # check original against new value
        is_good_val = False
        if _bool_is_same(new_tup[1], extrema_tup[1]) or new_tup[0] in extrema_tup:
            is_good_val = True

    return is_good_val


def to_exact_indexing(window, timeseries):
    """
    Convert create exact indexing from slicing of index
    Parameters
    ----------
    window : str, tuple, window, pd.Period
    timeseries : pd.Series

    Returns
    -------
    exact_window : tuple of pd.Timestamp
        the exact indexing of the input window (second resolution)
    returns None if there is no data in timeseries for window

    """
    if isinstance(window, (tuple, list)):
        check_length(window, 2)
        start, end = window
        # if window is in the correct format just return it
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            start_before_end(start, end)
        if start is None:
            return None
        # begin conversions/tests
        elif isinstance(start, str) and isinstance(end, str):
            if start == 'first':
                start = timeseries.index[0]
                end = make_end_of_day(pd.to_datetime(end))
            elif end == 'last':
                end = timeseries.index[-1]
                start = pd.to_datetime(start)
            else:
                start = pd.to_datetime(start)
                end = make_end_of_day(pd.to_datetime(end))
    elif isinstance(window, pd.Period):
        start = window.to_timestamp(how='s')
        end = window.to_timestamp(how='e')
    elif isinstance(window, int):
        start = timeseries.index[0]
        end = start + pd.Timedelta(days=window)
        end = make_end_of_day(end)

        # match data resolution
    subset = timeseries[start:end]
    if not subset.empty and len(subset) > 2:
        idx = random.randint(1, len(subset)-1)
        time_between_data = subset.index[idx]-subset.index[idx-1]
        window_res = str(time_between_data.components.minutes)+'T'
        if time_between_data.components.minutes == 0:
            window_res = str(time_between_data.components.seconds)+'s'

        exact_window = (start.ceil(window_res), end.floor(window_res))
    else:
        exact_window = None
    return exact_window


def make_end_of_day(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.replace(hour=23, minute=59, second=59)


def timestamp_to_decimal_hours(timestamp):
    decimal = ((timestamp.minute * 60)+timestamp.second)/(60*60)
    return timestamp.hour + decimal


def timedelta_to_hours(dt):
    return (dt.days * 24) + (dt.seconds / (60*60))

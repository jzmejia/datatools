"""
"""
import re
import pandas as pd
from typing import Union, Optional, Tuple

FrameOrSeries = Union[pd.DataFrame, pd.Series]


def parse_duration(duration: str) -> Tuple[float, str]:
    """parse a pandas duration string into int and str identifier

    Args:
        duration (str): pandas recognizable duration string.
            e.g. '1H'= 1 hour, '30T' = 30 minutes

    Returns:
        Tuple[int, str]: integer number 
    """
    num = re.findall(r'\d\.\d', duration)
    if not num:
        num = re.findall(r'\d', duration)
        if not num:
            raise ValueError(
                f'duration string {duration} not recognized'
            )
    unit = re.findall(r'\D', duration)
    return num[0], unit[0]


def c_rolling(data: FrameOrSeries,
              window: str,
              window_func='mean',
              min_periods: Optional[int] = None,
              win_type: Optional[str] = None
              ) -> FrameOrSeries:
    """apply and center datetime index of pandas rolling window function

    Args:
        data (pd.Series, pd.DataFrame): pandas datetime index obj.
        window (str): pandas recognized time string
        window_func (str): rolling window function to apply.
            Defaults to mean (rolling average)
        min_periods (int): Minimum number of observations in window 
            required to have a value. Defaults to None.
        win_type (str): Window type for rolling window calculation.
            If None, all points are evenly weighted. Defaults to None.

    Returns:
        Series or DataFrame: rolling window function with centered index
    """
    rolled = getattr(data.rolling(
        window, min_periods=min_periods, win_type=win_type), window_func)()
    rolled.index = rolled.index - (pd.Timedelta(window) / 2)
    return rolled


def add_in_quadrature(*num):
    total = 0
    for n in num:
        total += n ** 2
    return np.sqrt(total)

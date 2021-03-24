from pathlib import Path
import re

import pandas as pd
from datetime import timedelta, datetime
# import numpy as np
import scipy
from scipy import stats, signal

from datatools.utils import c_rolling


"""
melt_model.py

w.e. - water equivalent
    the volume of water that would be obtained from melting snow or ice.
    a value in meters w.e. = volume of water (m^3) / surface area (m^2)

Calculating melt rate
    Returns
        melt_rate (float) : units mm w.e. per hour



MoVE Weather Station Specific Information
Weather Stations:
    LOWC - Low Camp Weather Station
        dates active:
        NOTE:
            2018: no rain gauge & only an upward pointing rad sensor
    HIGH - High Camp Weather Station
        2018: no rain gauge & only a downward pointing rad sensor

Errors/Data Gaps/etc.
    Solar Radiation:
        Station LOWC - 2017 and station HIGH in 2017
        Data sets where a shadow is cast on sensors resulting in
        diurnal measurement drops.
        LOWC 12:00 - 13:30 UTC
        HIGH 14:45 - 15:30
             10:30 - 12:15


"""


def read_hobo_csv(csv_file,
                  all_columns=False,
                  skiprows=1,
                  index_col=1,
                  parse_dates=True,
                  *args,
                  **kwargs
                  ):
    """
    Reads data from a csv file exported from HOBOware.

    Parameters
    ----------
    csv_file : string
        A string containing the file name of the csv file to be read.
    all_columns : boolean (optional)
        Determines whether to read in all columns or just ones that we
        search for and relabel
        (RH, DewPt, Abs Pres, Temp, Attached, Stopped, Connected, EOF,
        Cond High Range,
        Cond Low Range).

        Default = False

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing data from HOBO csv file.
    """
    df = pd.read_csv(csv_file,
                     skiprows=skiprows,
                     index_col=index_col,
                     parse_dates=parse_dates,
                     na_values=['-888.88', '-888.9'])
    # Convert column names into something nicer
    columns = df.columns
    old_columns = columns
    rename_dict = {}
    solar_count = 1
    """
    (Rewrote Matts code)
    Note to Matt : If you want to search for a term and replace a column
    name with something different
    use python tuples instead, example:
    new_names = (('find_this', 'replace_with_this'),)

    use it like this:

    for old, new in  new_names:
        if old in label:
            new_name = new
            wantcol = True
    """
    new_names = ['RH', 'Gust', 'Wind Speed',
                 'Wind Direction', 'DewPt', 'Abs Pres', 'Rain', 'Temp']
    # cols = (old_columns, new_names)

    del df['#']
    for label in columns:
        new_name = label

        for name in new_names:
            if name in label:
                new_name = name

        # Account for multiple Solar Radiation Sensors and name them differently.
        # NOTE: current code only allows for up to two solar rad sensors <JZM>
        if 'Solar' in label:
            if solar_count == 1:
                new_name = 'Solar1'
                solar_count = 2
            elif solar_count == 2:
                new_name = 'Solar2'
                solar_count = 3
            else:
                print(">2 Solar Rad Sensors Detecteds")

        rename_dict[label] = new_name
    df = df.rename(columns=rename_dict)

    return df


def read_and_rename_hobo(file, tz_info='UTC'):
    """MoVE weather station specific df naming

    Args:
        file ([type]): [description]
        tz_info (str, optional): [description]. Defaults to 'UTC'.

    Returns:
        [type]: [description]
    """
    # print("reading in data from: {}".format(file))
    df = read_hobo_csv(file)
    file = file.name if isinstance(file, Path) else file
    # Determine and rename incoming and reflected solar data
    if 'Solar2' in df:
        if df['Solar1'].mean() < df['Solar2'].mean():
            df = df.rename(columns={'Solar1': 'Reflected',
                                    'Solar2': 'Solar'})
        elif df['Solar1'].mean() > df['Solar2'].mean():
            df = df.rename(columns={'Solar1': 'Solar',
                                    'Solar2': 'Reflected'})

        # df['Solar_corrected'] = scipy.ndimage.filters.median_filter(
        #     df['Solar'], 10)

    elif df.index[0].year == 2018:
        if 'LOWC' in file:
            df = df.rename(columns={'Solar1': 'Solar'})
            # df['Solar_corrected'] = scipy.ndimage.filters.median_filter(
            #     df['Solar'], 10)
        elif 'HIGH' in file:
            df = df.rename(columns={'Solar1': 'Reflected'})

    df = df.rename(columns={'Wind Speed': 'Wind_speed'})
    df.tz_localize(tz_info)
    return df


def read_gcnet_headers(file):
    """get data headers from GC-NET data file

    GC-NET Data files have the data headers on individual lines
    separated from the data with a blank line "\n

    Args:
        file ([type]): [description]

    Returns:
        [type]: [description]
    """
    f = open(file, 'r')
    headers = []
    while True:
        line = f.readline()
        if 'Data' in line:
            continue
        if line == '\n':
            break
        line = re.sub('^\d |^\d\d ', '', line)
        headers.append(line.rstrip())

    return headers


def rename_gcnet_headers(headers):
    col_names = []
    for header in headers:
        match = False

        # gcnet.parameter has no duplicates so if match is found
        # new column name for that header is param (edit to change)
        # and use gcnet[gcnet.parameter == param] to pull another
        # column that matches the header
        for param in gcnet.parameter:
            if param in header:
                col_names.append(param)
                match = True
                continue

        # exit if unique name already assigned to header
        if match:
            continue

        for unit in gcnet.unit.dropna():
            if unit in header:
                matching_df = gcnet[gcnet.unit == unit]
                matching_df = matching_df.drop(
                    matching_df[matching_df.parameter.isin(col_names)].index)
                if len(matching_df.parameter) > 1:
                    col_names.append(matching_df.iloc[0].parameter)
                else:
                    col_names.append(matching_df.parameter)
                match = True
                break

        if match is False:
            raise ValueError(
                f'gcnet header {header} could not be identified'
            )
    # check all headers were named
    if len(col_names) != len(headers):
        raise ValueError(
            f'column names not assigned for all headers'
        )
    return col_names


def read_JAR1_data(file):
    """Read in GC-NET weather data

    Arguments:
        file {str} -- file path

    Returns:
        jar1 {dataframe} -- weather data for station
                            index: date

    """
    col_names = rename_gcnet_headers(read_gcnet_headers(file))
    jar1 = pd.read_csv(file,
                       sep=' ',
                       skiprows=len(col_names)+2,
                       names=col_names,
                       na_values=['999.0000', '999.0'],
                       parse_dates=['Year'],
                       keep_date_col=True
                       )

    jar1 = jar1.set_index(jar1.Year
                          + jar1['Julian Decimal Time'].apply(lambda x: pd.Timedelta(days=x-1)))
    jar1['Avg Air Temp'] = jar1[jar1.columns[jar1.columns.str.contains('Air')]].mean(
        axis=1)
    jar1.index = jar1.index.round(freq='H')

#     P_atm_mbar = jar1.Atm_Pressure.resample('15T').pad()
#     df2 = pd.DataFrame(P_atm_mbar)
#     #* Convert mbar to ft H2O and m H2O
#     df2['P_atm_ftH2O'] = df2.Atm_Pressure * mbar2ftH2O
#     df2['P_atm_mH2O'] = df2.Atm_Pressure * mbar2mH2O
    return jar1


class WeatherStation:
    def __init__(self,
                 data,
                 read_with=read_and_rename_hobo,
                 default_albedo=0.7,
                 name=None
                 ):
        """a class for weather station data

        Args:
            data (pd.DataFrame): Meterological observations (timeseries)
            read_with (function): function to read in data with.
                Defaults to read_and_rename_hobo, used with HOBO/ONSET 
                weather station output .csv files.
            default_albedo (float, optional): Albedo value to use if
                it can not be calculated from weather station data.
                Defaults to 0.7.
            name (str, optional): weather station name. Defaults to None.



        Class Attributes:
        self.data: pd.Dataframe of weather station data
        self.name
        self.temperature : deg C
        self.relative_humidity
        self.solar
        self.reflected
        self.wind_direction
        self.wind_speed
        self.gust
        self.rain
        self.default_albedo : default value: 0.7

        NOTE: if data is not a HOBO weather station with recognized names
        you must set class attributes manually

        instance = WeatherStation(df,name='station name')
        instance.temperature = self.data['name of temperature col in df']



        """
        self.data = data if isinstance(
            data, pd.DataFrame) else read_and_rename_hobo(data)
        self.name = name
        self.temperature = self.get_data_with('Temp')
        self.relative_humidity = self.get_data_with('rh')
        self.solar = self.data['Solar'] if 'Solar' in self.data else None
        self.solar_corrected = None
        self.reflected = self.data['Reflected'] if 'Reflected' in self.data else None
        if 'Wind Direction' in self.data:
            self.wind_direction = self.data['Wind Direction']
            self.data = self.data.rename(columns={'Wind Direction':
                                                  'Wind_direction'})
        self.wind_speed = self.data['Wind_speed'] if 'Wind_speed' in self.data.columns else None
        self.gust = self.data['Gust'] if 'Gust' in self.data.columns else None
        self.rain = self.data['Rain'] if 'Rain' in self.data.columns else None
        self.units = 'temperature - Deg C\n Melt: mm mw h^-1'
        self.default_albedo = default_albedo
        self.shadow_error = pd.DatetimeIndex([])

    # def get_item_with(self, col_names, has):
    #     """Return column name containing string has

    #     Args:
    #         col_names (list of strings):
    #         has (str): [description]

    #     Returns:
    #         column containing has (str):
    #     """

    #     for col in col_names:
    #         if has.lower() in col.lower():
    #             break
    #     return col if has.lower() in col.lower() else None

    def get_data_with(self, contains_string):
        col_name = None
        for col in self.data.columns:
            if contains_string.lower() in col.lower():
                col_name = col

        return self.data[col_name] if col_name is not None else None

    def __repr__(self):
        return 'weather data object'

    def __str__(self):
        return 'Weather data from {} to {}'.format(
            self.temperature.index[0], self.temperature.index[-1])

    def apply_shadow_correction(self,
                                shaded_time,
                                **kwargs):
        """correct incoming solar radiation timeseries for shadow.

        remove spurious radiation drops assocaited with a shadow cast
        on the instruments. linearly interpolate over the bad data
        points to preserve a complete timeseries. creates a new
        class attribute 'solar_corrected'

        Args:
            shaded_time (Tuple(str,str)): string tuple of start time
            and end time corresponding to the daily shadow occurance.
            extend the time range to cover the good measurements on
            either side of the drop in irradiance from the shadow.

        Example:
            If a shadow is cast on the instrument every day (err on the
            side of a longer timespan) from 11:00-13:15, expand the
            window so it includes 'good' observations and covers the
            changing sun patterns.

            WeatherStation.apply_shadow_correction(('10:30','14:00'))

            now your instance of WeatherStation will have an attribute
            self.solar_corrected -> pd.series

        """
        data_shaded = self.solar.between_time(shaded_time[0], shaded_time[1])
        for day in data_shaded.index.to_period('D').unique():
            shaded = data_shaded[day.start_time: day.end_time]
            self.shadow_error = self.shadow_error.append(
                shaded[shaded < min(shaded[0], shaded[-1])].index)

        # set shaded observations to None and linearly interp to fill
        self.solar_corrected = self.solar
        self.solar_corrected.loc[self.shadow_error] = None
        self.solar_corrected = self.solar_corrected.interpolate(
            method='linear')
        self.data['solar_corrected'] = self.solar_corrected
        pass

    def calc_albedo(self, daily=True):
        # check if there is incoming and reflected solar radiation
        if self.solar is not None and self.reflected is not None:

            incoming = self.solar_corrected if self.solar_corrected is not None else self.solar
            df = c_rolling(incoming, '2H').to_frame(name='incoming')
            df['outgoing'] = c_rolling(self.reflected, '2H')
            df.dropna(how='any', inplace=True)
            incoming = df.incoming
            outgoing = df.outgoing

            self.albedo = outgoing / incoming

            if daily:
                albedo = []
                for name, group in df.resample('D'):
                    if group.empty is False:
                        albedo.append({'date': name, 'albedo':
                                       group['outgoing'][group['incoming'].idxmax(
                                       )] / group['incoming'].max()})

                self.albedo = pd.DataFrame(albedo)
                self.albedo.set_index('date', inplace=True)
                self.albedo = self.albedo.albedo

            self.albedo.loc[self.albedo >= 1] = None
            self.albedo = self.albedo.fillna(method='ffill')
            self.data['albedo'] = self.albedo
            self.data['albedo'] = self.data['albedo'].fillna(method='ffill')

        else:
            self.albedo = self.default_albedo
            print(
                f'{self.name} weather station, {self.data.index[0]}'
                f' - {self.data.index[-1]}\n'
                f'Using default abledo {self.default_albedo} for'
                f' melt rate calculations.')

        return self.albedo

    def calc_melt(self,
                  incoming_shortwave_radiation=None,
                  threshold_temp=0):
        '''
        Calculate hourly melt rates (mm m.w. equivalent h^-1)
        using the enhanced temperature-index glacier melt model


        M = {   TF * T + SRF * (1 - alpha) * G  if T >  TT
                0                               if T <= TT
        where
            M - Melt Rate (mm per hour)
            TF - Temp Factor (mm h^-1 C^-1)
            T - mean air Temp of each time step (C)
            SRF - Shortwave radiation factor (m^2 mm W^-1 h^-1)
            alpha - albedo
            G - incoming shortwave radiation (W m^-2)
            TT - threshold Temp (0 deg C)

        Ref:
        ----
        Pellicciotti et al. (2005). An enhanced temperature - index
            glacier melt model including the shortwave radiation
            balance: development and testing for Haut Glacier
            d'Arolla, Switzerland. J. Glac. 51(175), 573-587.

        Parameters
        ---------
        temperature :
            Air Temperature
            Units: degree C

        incoming_shortwave_radiation :
            Incoming shortwave solar radiation
            Units: W m^-2
        albedo :


        temperature_threshold :
            temperature at or below no melt will occur
            deg C
            default value = 0


        Output
        ------
            df : pd.dataframe
                data frame (time-indexed)
                melt rates: mm w.e. h^-1

            temp_threshold : optional, int
                determines the cut off for melt to occur
                default = 0 deg C
            ice_snow_transition : optional, string
                date string in a format readable by pd.to_datetime()
                if ice_snow_transition=2017, '2017-08-17 00:00:00' is used
                if ice_snow_transition=None, no transition is given
                    bare ice albedo used for entire calc period

        '''
        # Define constants:
        TEMPERATURE_FACTOR = 0.05
        SOLAR_RADIATION_FACTOR = 0.0094

        if self.solar is None:
            raise ValueError(
                'An incoming solar radiation timeseries is required '
                'to calculate meltwater production.\n Assign values to '
                'WeatherStation.solar before executing .calc_melt()'
            )

        # create a df with temperatures above threshold (melt not 0)
        # then add incoming solar radiation and albedo values for
        # melt_rate calculation
        df = self.temperature[self.temperature >
                              threshold_temp].to_frame(name='temperature')
        df['incoming'] = c_rolling(
            self.solar_corrected, '2H') if self.solar_corrected is not None else c_rolling(self.solar, '2H')

        self.calc_albedo(daily=True)
        if isinstance(self.albedo, float):
            df['albedo'] = self.albedo
        else:
            albedo = pd.DataFrame(data={
                'start_day': df.groupby(df.index.date).apply(
                    lambda x: x.index.min()),
                'albedo': self.albedo}).dropna(how='any')
            albedo = albedo.set_index('start_day')
            df['albedo'] = albedo
            df['albedo'] = df['albedo'].fillna(method='ffill')

        df['melt_rate'] = melt_equ(df.temperature, df.albedo, df.incoming)
        self.data['melt_rate'] = df['melt_rate']
        self.data['melt_rate'] = self.data['melt_rate'].fillna(0.)
        self.melt_rate = self.data['melt_rate']
        return self.melt_rate

    # @ property
    # def solar(self):
    #     return self._solar

    # @ solar.setter
    # def solar(self, value):
    #     if self.data['Solar'].isin(value).all() is False:
    #         self.data['Solar'] = value
    #     self._solar = value

    # @property
    # def solar_corrected(self):
    #     return self._solar_corrected

    # @solar_corrected.setter
    # def solar(self, value):
    #     if self.data['Solar_corrected'].isin(value).all() is False:
    #         self.data['Solar_corrected'] = value
    #     self._solar_corrected = value

    # @ property
    # def reflected(self):
    #     return self._reflected

    # @ reflected.setter
    # def reflected(self, value):
    #     if self.data['Reflected'].isin(value).all() is False:
    #         self.data['Reflected'] = value
    #     self._reflected = value


def melt_equ(temperature,
             albedo,
             incoming_solar_radiation,
             TEMPERATURE_FACTOR=0.05,
             SOLAR_RADIATION_FACTOR=0.0094):
    """[summary]

            M = {   TF * T + SRF * (1 - alpha) * G  if T >  TT
            0                               if T <= TT
    where
        M - Melt Rate (mm per hour)
        TF - Temp Factor (mm h^-1 C^-1)
        T - mean air Temp of each time step (C)
        SRF - Shortwave radiation factor (m^2 mmW^-1 h^-1)
        alpha - albedo
        G - incoming shortwave radiation (W m^-2)
        TT - threshold Temp (0 deg C)

    Args:
        temperature (float): mean air Temp of each time step (C)
        albedo (float): [description]
        incoming_solar_radiation (float): [description]
        TEMPERATURE_FACTOR (float, optional): [description].
            Defaults to 0.05.
        SOLAR_RADIATION_FACTOR (float, optional): [description].
            Defaults to 0.0094.

    Returns:
        melt_rate: mm melt water equivelent per hour
    """

    return ((TEMPERATURE_FACTOR * temperature)
            + (SOLAR_RADIATION_FACTOR
                * (1 - albedo)
                * incoming_solar_radiation))


# create df for GC-NET weather station naming conventions
parameters = [
    'Station Number',
    'Year',
    'Julian Decimal Time',
    'SW_down',
    'SW_up',
    'Net Radiation',
    'TC Air 1',
    'TC Air 2',
    'CS500 T Air 1',
    'CS500 T Air 2',
    'RH 1',
    'RH 2',
    'U1',
    'U2',
    'U Dir 1',
    'U Dir 2',
    'Atmos Pressure',
    'Snow Height 1',
    'Snow Height 2',
    'T Snow 1',
    'T Snow 2',
    'T Snow 3',
    'T Snow 4',
    'T Snow 5',
    'T Snow 6',
    'T Snow 7',
    'T Snow 8',
    'T Snow 9',
    'T Snow 10',
    'Battery Voltage',
    'U 2m from theory',
    'U 10m from theory',
    'Height of profile 1',
    'Height of profile 2',
    'Albedo',
    'Peak wind speed',
    'Zenith Angle',
    'QC identifier col. 1',
    'QC identifier col. 2',
    'QC identifier col. 3',
    'QC identifier col. 4',
    'SWinMax',
    'SWoutMax',
    'NetRadMax'
]

index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', None,
         'AK', 'AL', 'AM', 'AN', None, None, None]
units = ['01 .. 18',
         '1995 – 1999',
         '0.0000 – 365.9583',
         '[W m-2]', '[W m-2]', '[W m-2]',
         'Air Temperature [°C]', 'Air Temperature [°C]',
         'Air Temperature [°C]', 'Air Temperature [°C]',
         'Relative Humidity [%] **', 'Relative Humidity [%] **',
         'Wind Speed [m/s]', 'Wind Speed [m/s]',
         'degrees [0-360]', 'degrees [0-360]',
         '[mb]',
         '[m]', '[m]',
         '[°C]', '[°C]', '[°C]', '[°C]', '[°C]', '[°C]',
         '[°C]', '[°C]', '[°C]', '[°C]',
         'VDC',
         'Wind Speed [m/s]', 'Wind Speed [m/s]',
         'm', 'm', None, None, '[deg]', None, None, None, None,
         '[W m-2]', '[W m-2]', '[W m-2]',
         ]
gcnet = pd.DataFrame(data={'parameter': parameters,
                           'letter index': index,
                           'unit': units
                           })



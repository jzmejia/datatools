"""
hydrotools.py
Author: JZMejia
Last Update: 2020 Apr 18

Post process mouin water level and stream stage data
data collection specifications. All data is read using pd.from_csv()
with arguments based off of campbell data logger output files.
Raw Data :
    Data Logging :  Campbell Scientific
                    CR-1000 Data Logger
                    AVW200 vibrating wire module
                    CURS100 100 Ohm current shunt terminal input module
    moulin water level
        Geokon Vibrating Wire Piezometer
        Model: 4500 HD
    stream stage
        Global Water - Xylum
        WL705 Ultrasonic Water Level Sensor
        WL705-012 range:  4"-12'
        WL705-048 range: 15"-48'
        
        
        
        
        
Geokon 4500-HD Piezometer Read Out Info
    Temperatures:
        Piezometer Thermistor:
            Range: -80 to +150 deg C
            Accuracy: +/- 0.5 deg C
            
        Resistance to Temperature Equation
        T = 1 / (A + B (LnR) + C (LnR ** 3)) - 273.15
            T: Temperature in C
            LnR: Natural Log of Thermistor Resistance
            A = 1.4051 * 10e-3
            B = 2.3690 * 10e-4
            C = 1.0190 * 10e-7
        TR
        TT
        
        Temperature Correction (P_T)
        P_T = (T1 - T0) * K
        T1: Current Temperature
        T0: Initial Zero Temperature
        K: Thermal Factor
        
            long cables can be applied ~48.5 Ohm per km
            (14.7 Ohms per 1000') at 20 deg C
            
        

    Pressure
        Pressure Calculation:
            digits = (1 / Period (seconds)) ** 2 * 10e-3 
            or
            digits = Hz ** 2 / 1000
        P = (R1 - R0) * G
            R1: Current Reading
            R0: Initial Zero Reading
            G: Linear Calibration Factor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from .units import convert
from .melt_model import read_JAR1_data

from .constants import ICE_DENSITY, WATER_DENSITY


def masl_to_pctoverburden(waterLvl, flotation_mbed, bed_elev):
    """
    convert water level from m asl to fraction of overburden pressure.

    Parameters
    ----------
    waterLvl : float
        water level in meters above sea level
        flotation_range : tuple, float
            range of flotation water levels (m asl)
            **Output from calc_flotation**
    Returns
    -------
        pct_overburden : tuple, float
        range of overburden pressures for given water level and floations
    """
    waterLvl_mbed = (waterLvl-bed_elev[0],
                     waterLvl-bed_elev[1])
    pct_overburden = tuple(
        map(lambda x, y: x/y, waterLvl_mbed, flotation_mbed))
    return pct_overburden


def gwl2fob(gwl, ice_thickness, surface_elevation):
    '''convert gwl to fob

    Parameters
    ----------
        gwl: float
            ground water level altitude (m asl)
        ice_thickness : float
            ice thickness in meters
        surface_elevation: float 
            surface elevation in meters above sea level
    Returns
    -------
        fob : float
            fraction of overburden pressure
    '''
    bed_elevation = surface_elevation - ice_thickness
    water_level_above_bed = gwl - bed_elevation
    return wlb2fob(water_level_above_bed, ice_thickness)


def wlb2fob(water_level_above_bed, ice_thickness):
    '''Convert water level above bed to fraction of ice overburden pressure

    using the densities of ice and water of 917 and 1000 kg/m^3
    respectively
    
    Parameters
    ----------
        water_level_above_bed : float
            water level in meters above base/bed
        ice_thickness : float
            ice thickness in meters

    Returns
    -------
        fob : float
            fraction of ice overburden pressure
    '''
    return (water_level_above_bed
            / ((ICE_DENSITY/WATER_DENSITY) * ice_thickness))



def convert_axis_to_fob(ax_gwl, ax_fob, ice_thickness, surface_elev):
    '''update right plot yaxis to fraction of overburden pressure

    update subplot right (second) axis to fraction of overburden 
    pressure corresponding to existing values on left axis

    Parameters
    ----------
    ax_gwl : matplotlib.pyplot axis object
        axis with data plotted as ground water level or (water level) in meters
    ax_fob : axis object
        opposite axis object to update as fraction of overburden pressure
    ice_thickness : float
        ice thickness in meters
    surface elevation : float
        surface elevation in meters
    '''
    y1, y2 = ax_gwl.get_ylim()
    ax_fob.set_ylim(gwl2fob(y1, ice_thickness, surface_elev),
                    gwl2fob(y2, ice_thickness, surface_elev))
    ax_fob.figure.canvas.draw()
    pass


# zero_reading_time
# zero_reading_value

'''
P0 = atmospheric pressure (mH20) corresponding with
 piezometer zero reading
JEME:
    datetime: '2017-07-20 21:15:00'
    submerged depth: 21.31 m
    P0 time: '2017-07-20 21:00:00'
    P0 val: 906.1 mbar
            9.24222 mH2O

RADI:
    datetime: '2017-07-29 20:30:00'
    submerged depth: 11.820 m
    P0 time: '2017-07-29 19:59:31.2'
    P0 val: 918.3 mbar
            9.3666 mH2O

'''
# Atmospheric Pressure at time of instrumentation (zero reading)
P0_jeme = 9.24222
P0_radi = 9.36666

# # Read in GC-NET JAR1 Weather Data
# # concat data, upsample atm pressure data & convert units
# jar1a = read_JAR1_data(
#     '../B_WEATHER-STATIONS/JAR/JAR1_WEA_0012017_1232018.txt')
# jar1b = read_JAR1_data(
#     '../B_WEATHER-STATIONS/JAR/JAR1_WEA_1212018_1252019.txt')
# JAR1 = pd.concat([jar1a, jar1b], sort=False, join='outer')
# # remove duplicates
# JAR1 = JAR1[~JAR1.index.duplicated(keep='first')]
# # Convert atmospheric pressure from mbar to m h2o
# # 1 mbar = 0.0102 m H2O
# # and upsample to 15 minutes to match moulin data
# P_atm = pd.DataFrame(convert('mbar', 'mH2O', JAR1['Atm_Pressure']))
# P_atm = P_atm.resample('15T').nearest()


def read_cr1000_csv(file, drop_cols=['RECORD', 'BattV', 'Freq', 'Amp', 'SNRat',
             'NFreq', 'DRat', 'Digits']):
    '''read in data file generated from cr1000 data logger
    
    Parameters
    ----------
        file : str
            csv file name with cr1000 data
        drop_cols : list of column names to drop
            Defaults to: ['RECORD', 'BattV', 'Freq', 'Amp', 'SNRat',
             'NFreq', 'DRat', 'Digits']

    Returns
    -------
        df : pd.DataFrame
    '''

    df = pd.read_csv(file, index_col=0, skiprows=[0, 2, 3],
                     parse_dates=True, na_values="NAN")
    df.drop(drop_cols,
            inplace=True, axis=1)
    return df


def read_str17_csv(file):
    df = pd.read_csv(file, index_col=0, skiprows=[0, 2, 3],
                     parse_dates=True, na_values="NAN")
    df.drop(['RECORD'], inplace=True, axis=1)
    df = df.dropna()
    return df


def calc_stage_radi(file, ref_datum=4, bad_data_readout=67, data_header='WL705012'):
    """calculate stream stage from ultrasonic water level measurements

    read in data file with ultrasonic water level measurements and convert
    to stream stage in meters

    Parameters
    ----------
        file : str
            data file name to be read in
        ref_datum : float, int
            arbitrary datum to use to calculate stream stage in meters. 
            Use distance from sensor to stream base if known. Defaults to 4 m.
        bad_data_readout : float, int
            value to use to filter observations, typically this value is the maximum
            measurement distance able to be recorded by the sensor. Defaults to 67 for
            Global Water WD705-012 ultrasonic water level sensors. The sensor will record this
            maximum value when a return signal is not reflected back to the sensor as in the 
            case of turburlent water flow, or other environmental sources.
        data_header : str
            name of column in .csv file with stream stage data. 
            Defaults to WL705012 (named after sensor)

    Returns
    -------
        pd.DataFrame, np.float64
            dataframe with stream stage in meters above arbitrary datum and distance
            between sensor and water surface in meters. Data is indexed with
            the timestamp of each measurement. 
    """
    logger_to_stream = read_str17_csv(file)
    logger_to_stream = logger_to_stream[logger_to_stream['data_header'] < bad_data_readout]
    lts_m = convert('in', 'm', logger_to_stream['data_header'])
    stream_stage = ref_datum - lts_m
    df = pd.DataFrame({'stage': stream_stage,
                       'logger_to_stream': lts_m})
    return df.astype(np.float64)


def calc_stage_jeme(file,drop_list,ref_datum=4,water_sensor='WL705048',ablation_sensor='WL705012',bad_data_readout=80):
    """calculate stream stage from distance ranger measurements of water and ground surface

    Parameters
    ----------
    file : str
    drop_list : List(str)
        list of timestamps corresponding to datapoints to be dropped
        from dataset (or replaced with NaNs)

    Returns
    -------
    pd.DataFrame

    """
    logger_to_stream = pd.read_csv(file, skiprows=[0, 2, 3], index_col=0,
                                   parse_dates=True)
    logger_to_stream = logger_to_stream[logger_to_stream['water_sensor'] < bad_data_readout]
    ltg = convert('in', 'm', logger_to_stream['ablation_sensor'])
    lts = convert('in', 'm', logger_to_stream['water_sensor'])
    stream_stage = ref_datum - lts
    for timestamp in drop_list:
        stream_stage[timestamp] = np.nan

    df = pd.DataFrame({'stage': stream_stage,
                       'logger_to_stream': lts,
                       'logger_to_ground': ltg})
    df.index = pd.to_datetime(df.index)
    df = df.astype(np.float64)
    return df


def calc_wlb_radi(file, P_atm, depth_to_water, ice_thickness, ice_surface_elevation, zero_reading_value):
    """
    * NOTE: While the unit marker in the input file for the moulin says
        head is in feet, it is actually in meters,
        CONFIRMED by JZM via datalogger program 2019
    Parameters
    ----------
        file : str
            .csv file from data logger for radical moulin 2017 data
        P_atm : atmospheric pressure timeseries
        depth_to_water : float
            depth from ice surface to water in moulin at the time 
            of instrumentation. Units are in negative meters.
        ice_thickness : float
            ice thickness in meters
        ice_surface_elevation : float
            ice surface elevation in meters above sea level
        zero_reading_value: float
            Barometric pressure at the time of instrumentation in meters of water
        piezometer_lowering_adjustments
    Output
        water_level_above_bed:
            water level within moulin in meters above the bed
        submerged_depth:
            submerged depth of sensor in meters
    """

    df = pd.read_csv(file, skiprows=[0, 2, 3],
                       na_values="NAN", index_col=0,
                       parse_dates=True)
    

    # Set location specific constants
    # TODO: use updated dictionary in constants.py
    depth_to_water = -244.38  # (m)
    ice_thickness = 712  # (m)
    ice_surface_elevation = 933.2  # m asl
    bed_elevation = ice_surface_elevation - ice_thickness
    
    n = len(df.Lvl)

    wlb1 = ice_thickness + depth_to_water
    Z_array = np.empty(n, dtype=object)


    wlb2 = wlb1 - 16.853
    wlb3 = wlb2 - 70
    wlb4 = wlb3 - 4.05
    wlb5 = wlb4 - 1
    wlb6 = wlb5 - 4.6

    Z_array[:881] = wlb1
    Z_array[881:920] = wlb2
    Z_array[920:1705] = wlb3
    Z_array[1705:1713] = wlb4
    Z_array[1713] = wlb5
    Z_array[1714:] = wlb6

    # correct for atmospheric pressure
    df['P_atm'] = P_atm[str(df.index[0]):str(df.index[-1])]
    df['Change_in_Patm'] = df['P_atm'] - zero_reading_value
    df['Lvl_corrected'] = df.Lvl - df.Change_in_Patm

    water_level_above_bed = df.Lvl + Z_array
    water_level_above_bed_corrected = df.Lvl_corrected + Z_array
    gwl = water_level_above_bed + bed_elevation
    gwl_corrected = water_level_above_bed_corrected + bed_elevation
    fob = wlb2fob(water_level_above_bed, ice_thickness)

    df = pd.DataFrame({'water_level_above_bed': water_level_above_bed,
                       'water_level_above_bed_corrected':
                       water_level_above_bed_corrected,
                       'submerged_depth': df.Lvl,
                       'submerged_depth_corrected': df.Lvl_corrected,
                       'water_level_above_bed_fob': fob,
                       'ground_water_level_altitude': gwl,
                       'ground_water_level_altitude_corrected': gwl_corrected})
    # clean output
    df = df.drop(df['2017-08-04 19:00':'2017-08-04 19:40'].index)
    return df.astype(np.float64)


def calc_submerged_depth(df, zero_reading):
    df['submerged_depth'] = -convert('ft', 'm', df.Lvl) + zero_reading
    return df


# class moulin_data:
#     def __init__(self,
#                  input_data = None,
#                  ice_thickness = None,
#                  ice_surface_elevation = None,
#                  depth_to_water = None,
#                  bed_elevation = None
#                  ):
#         self.measurements = load_data(input_data)
#         self.sampling_rate = None
# def load_data(data):
#     if is_instance()

def calc_wlb_pira(file_3sec, file_15min, zero_reading):
    zero_reading = 1.1400000  # m
    # Determine sensor height to calc wlb
    ice_thickness = 503.
    #! in my notes it says hit water at 132 m
    #! zero reading 8815.66 digits
    #! a total of '180 m of cable in the moulin'
    depth_to_water = -143.5
    ice_surface_masl = 764.9
    # bed elevation (masl)
    bed_elevation = ice_surface_masl - ice_thickness
    # sensor depth upon lowering
    piz_depth = 11.

    # read in data files (lowering 3sec and monitoring 15 m)
    df_3s = read_cr1000_csv(file_3sec)
    df_15m = read_cr1000_csv(file_15min)

    df_3s = calc_submerged_depth(df_3s, zero_reading)
    df_15m = calc_submerged_depth(df_15m, zero_reading)

    # determine sensor height at the end of initial lowering 7/9/18
    piz_hgt0 = ice_thickness + depth_to_water - piz_depth
    # JUL 10 lowering adjustment
    piz_adj1 = piz_hgt0 - 3.048
    # JUL 12 lowering adjustment
    piz_adj2 = piz_adj1 - 3.01752
    # JUL 12 sensor drop
    piz_adj3 = piz_adj2 - 1.15824
    # JUL 14 matt lowering
    piz_adj4 = piz_adj3 - 0.51816
    # JUL 15 Drop 1
    piz_adj5 = piz_adj4 - 0.12192
    # JUL 15 Drop 2
    piz_adj6 = piz_adj5 - 0.97536
    # JUL 15 Lowering Adjustment
    piz_adj7 = piz_adj6 - 4.328
    print(piz_adj7, 'm')
    # calculate water level above bed
    from numpy import nan
    df_3s['piz_hgt'] = nan
    df_3s['piz_hgt']['2018-07-10 12:00:00':'2018-07-10 15:14:27'] = piz_hgt0
    df_3s['piz_hgt']['2018-07-10 15:20:00':'2018-07-12 18:18:27'] = piz_adj1
    df_3s['piz_hgt']['2018-07-12 18:38:03':'2018-07-12 22:44:50'] = piz_adj2
    df_3s['piz_hgt']['2018-07-12 22:44:53':'2018-07-14 21:28:36'] = piz_adj3
    df_3s['piz_hgt']['2018-07-14 21:29:27':'2018-07-15 01:54:03'] = piz_adj4
    df_3s['piz_hgt']['2018-07-15 01:54:18':'2018-07-15 03:01:24'] = piz_adj5
    df_3s['piz_hgt']['2018-07-15 03:03:54':'2018-07-15 12:30:00'] = piz_adj6
    df_3s['piz_hgt']['2018-07-15 12:41:09':] = piz_adj7
    df_3s = df_3s.dropna()
    # calculate water level above bed
    df_3s['water_level_above_bed'] = df_3s['submerged_depth'] + df_3s['piz_hgt']
    # adjustment for 15 minute script
    df_15m['piz_hgt'] = piz_adj7
    df_15m['water_level_above_bed'] = df_15m['submerged_depth'] + \
        df_15m['piz_hgt']

    df = df_3s.append(df_15m, sort=False)
    df['water_level_above_bed_fob'] = wlb2fob(
        df['water_level_above_bed'], ice_thickness)
    # get names of indexes for which
    # water level above bed > ice surface
    # delete these rows from dataframe
    df.drop(index=df[df.water_level_above_bed > 503].index, inplace=True)
    df['ground_water_level_altitude'] = df['water_level_above_bed'] + bed_elevation
    # df = df_.astype(np.float64)
    return df


def calc_wlb_jeme(file, P_atm):
    """
    Script to read in, correct, and calculate head for JMM
    located at Low Camp
        Parameters__________________________________________
            JMM_file - Low Camp JMM moulin CR1000 file path
            -raw data----
            JMM.lvl - water level (meters H2O)
                JMM.time - timeseries, dtype = datetime64[ns]
            JMM.digits - raw digits (Freq^2/1000)
            (can use to calc height without temp correction
            this is the raw data from the piezometer)
        Returned____________________________
            df.index
                dtype: datetime64[ns], UTC
            df.water_level_above_bed
                dtype:  float64
                ref:    ice thickness = 503 m
                not corrected for P_atm (cuts off timeseries)
            df.water_level_above_bed_corrected:
                corrected for P_atm
            df.submerged_depth: submerged depth (m)
                dtype:  float64
                units:  meters of water
            df.submerged_depth_corrected
                corrected for P_atm
            df.water_level_above_bed_fob
                dtype: float64
                units: fraction of overburden Pressure
                        calculated using icethickness=503 m
            df.gwl (ground water level altitude)
                units: meters above sea level
                ref:   ocean = 0
            df.gwl_corrected (ground_water_level_altitude corrected)
                units: meters asl
                ref:   ocean = 0
                corrected for P_atm


    """
    ice_thickness = 503
    ice_surface_masl = 765.8
    depth_to_water = -201.31
    bed_elevation = ice_surface_masl - ice_thickness

    a = [0]
    a.extend(range(2, 3129))
    a.extend(range(3551, 4149))
    jeme = pd.read_csv(file, index_col=0, skiprows=a, na_values="NAN",
                       parse_dates=True)

    # sensor_level_above_bed + submerged_depth = wlb
    # define initial sensor level above bed and change
    #  for drops of sensor
    slb1 = ice_thickness + depth_to_water
    slb1a = slb1 - 5.5
    slb2 = slb1a - 138.16
    slb3 = slb2 - 4.8
    # hgt = timeseries of sensor level above
    hgt = np.empty(len(jeme.Lvl), dtype=float)
    hgt[0:69] = slb1
    hgt[69:422] = slb1a
    hgt[422:812] = slb2
    hgt[812:] = slb3
    jeme = jeme.assign(Hgt=hgt)

    # Correct measured water level for changes in atmospheric pressure
    jeme['P_atm'] = P_atm[str(jeme.index[0]):str(jeme.index[-1])]
    jeme['Change_in_P_atm'] = jeme['P_atm']-P0_jeme
    jeme['Lvl_corrected'] = jeme.Lvl - jeme.Change_in_P_atm

    # calcualte head in meters above bed
    water_level_above_bed = jeme.Lvl + jeme.Hgt
    water_level_above_bed_corrected = jeme.Lvl_corrected + jeme.Hgt
    fob = wlb2fob(water_level_above_bed, ice_thickness)
    gwl = water_level_above_bed + bed_elevation
    gwl_corrected = water_level_above_bed_corrected + bed_elevation
    df = pd.DataFrame({'water_level_above_bed': water_level_above_bed,
                       'water_level_above_bed_corrected':
                       water_level_above_bed_corrected,
                       'water_level_above_bed_fob': fob,
                       'submerged_depth': jeme.Lvl,
                       'submerged_depth_corrected': jeme.Lvl_corrected,
                       'ground_water_level_altitude': gwl,
                       'ground_water_level_altitude_corrected': gwl_corrected})
    df = df.astype(np.float64)
    return df



def read_moulin18(file, all_data=None):
    # Read in raw 2018 CR1000 datalogger files and convert units to meters
    data = pd.read_csv(file, skiprows=[0, 2, 3], na_values="NAN",
                       index_col=0, parse_dates=True)
    data['submerged_depth'] = -convert('ft', 'm', data.Lvl)
    data.rename(columns={'Lvl': 'submerged_depth_ft'}, inplace=True)
    if all_data != True:
        columns = ['RECORD', 'BattV', 'PTemp_C', 'Freq', 'Amp', 'SNRat',
                   'NFreq', 'DRat', 'TR', 'TT', 'Digits']
        data.drop(columns, inplace=True, axis=1)
    if 'Stage' in data:
        data['logger_to_stream'] = convert('in', 'm', data.Stage)
        data['stage'] = 4 - data['logger_to_stream']
        data.rename(columns={'Stage': 'logger_to_stream_in'}, inplace=True)
    if 'Abl' in data:
        data['logger_to_ground'] = convert('in', 'm', data.Abl)
        data.rename(columns={'Abl': 'logger_to_ground_in'}, inplace=True)
    return data

# def utc_to_loc_wgris(data):
#     '''shift from utc to local time west greenland = utc-3'''
#     data.index = data.index - timedelta(hours=3)
#     return data.index



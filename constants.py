#!/user/bin/env python
"""
Constants
---------
    Physical
    GRAVITY: float
    GRAVITY_UNIT: str

    ICE_DENSITY: float
    WATER_DENSITY: float
    DENSITY_UNIT: str

    SECONDS_PER_DAY: int
    
    Glen's flow law parameters:
    A
    n

Station Stats
-------------
    LOWCAMP: Dict
    JEME: Dict
    PIRA: Dict
    RADI: Dict

"""


GRAVITY = 9.81
GRAVITY_UNIT = 'm s^-2'

# densities
ICE_DENSITY = 917.0
WATER_DENSITY = 1000.0
DENSITY_UNIT = "km m^-3"


GEOTHERMAL_FLUX = 0.06
GEOTHERMAL_FLUX_UNIT = 'W m^-2'

LATIENT_HEAT_OF_FUSION = 3.35E5
LATIENT_HEAT_OF_FUSION_UNIT = 'J kg^-1'

# time
SECONDS_PER_DAY = 60 * 60 * 24


LOWCAMP = {
    'surface elevation': 765.8,
    'ice thickness': 503,
    'uncertainty': 100,
}


JEME = {
    'year': 2017,
    'surface elevation': 765.8,
    'ice thickness': 503,
    'depth to water': -201.31,
    'zero reading': {
        'submerged depth': 21.31,
        'submerged depth unit': 'm',
        'submerged depth time': '2017-07-20 21:15:00',
        'atmospheric pressure': 9.24222,
        'atmospheric pressure unit': 'mH2O',
        'atmospheric pressure time': '2017-07-20 21:00:00'
    }
}

PIRA = {
    'year': 2018,
    'surface elevation': 764.9,
    'ice thickness': 503,
    'depth to water': -143.5,
    'pizometer depth': 11,
    'zero reading': {
        'logger': 1.14,
        'logger unit': 'm'
    }
}


RADI = {
    'surface elevation': 933.2,
    'ice thickness': 712,
    'depth to water': -244.38,
    'zero reading': {
        'submerged depth': 11.820,
        'submerged depth unit': 'm',
        'submerged depth time': '2017-07-29 20:30:00',
        'atmospheric pressure (P0)': 9.36666,
        'atmospheric pressure unit': 'meters of water',
        'atmospheric pressure time': '2017-07-29 19:59:31.2'
    }
}

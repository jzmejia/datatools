_prefix = {'y': 1e-24,  # yocto
           'z': 1e-21,  # zepto
           'a': 1e-18,  # atto
           'f': 1e-15,  # femto
           'p': 1e-12,  # pico
           'n': 1e-9,   # nano
           'u': 1e-6,   # micro
           'm': 1e-3,   # mili
           'c': 1e-2,   # centi
           'd': 1e-1,   # deci
           'k': 1e3,    # kilo
           'M': 1e6,    # mega
           'G': 1e9,    # giga
           'T': 1e12,   # tera
           'P': 1e15,   # peta
           'E': 1e18,   # exa
           'Z': 1e21,   # zetta
           'Y': 1e24,   # yotta
           }

# hlist = ['H', 'h', 'hrs', 'hours', 'hour']
# mlist = ['M', 'm', 'min', 'minutes', 'minute']
# dlist = ['D', 'd', 'days', 'day']
# slist = ['S', 's', 'seconds', 'sec', 'second']

# t_dict = {
#     'day': dlist,
#     'hour': hlist,
#     'minute': mlist,
#     'second': slist
# }

conversion_factors = {
    'mbar': {
        'mH2O': 0.0102,
        'ftH2O': 0.03346
    },
    'ft': {'m': 0.3048},
    'in': {
        'm': 0.0254,
        'mm': 25.4,
    }
}


def convert(from_unit: str, to_unit: str, value: float) -> float:
    conversion_factor = get_conversion_factor(from_unit, to_unit)
    return value * conversion_factor


def get_conversion_factor(from_unit: str, to_unit: str):
    conversion_factor = find_in_dict(from_unit, to_unit)
    if conversion_factor is None:
        conversion_factor = check_reverse(from_unit, to_unit)
    return conversion_factor


def check_reverse(from_unit, to_unit):
    if find_in_dict(to_unit, from_unit) is None:
        raise ValueError(
            f'Units {from_unit} and {to_unit} not found in conversion_factors'
        )
    else:
        return 1 / find_in_dict(to_unit, from_unit)


def find_in_dict(from_unit, to_unit):
    if from_unit in conversion_factors.keys():
        if to_unit not in conversion_factors.get(from_unit).keys():
            raise ValueError(
                f'Input unit {to_unit} not in conversion_factors dictionary/'
            )
        return conversion_factors.get(from_unit).get(to_unit)

# python
def is_in_range(
    value         : float,
    min           : float,
    max           : float,
    argument_name : str = '' 
):
    if not (min <= value and value <= max):
        raise Exception(
            f'value \'{argument_name}\' is not in range [{min}, {max}].'
        )
    

def is_in_range_no_include(
    value         : float,
    min           : float,
    max           : float,
    argument_name : str = '' 
):
    if not (min < value and value < max):
        raise Exception(
            f'value \'{argument_name}\' is not in range ({min}, {max}).'
        )
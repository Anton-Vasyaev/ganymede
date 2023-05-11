# python
from typing import Set, List, Dict, Any, TypeVar, Type, cast
# project
from .data.config_info import ConfigBlock
from .error import DarknetConfigLoadException


T = TypeVar('T')

DEFAULT_VALUES : Dict[Type, Any] = dict({
    int : 0,
    float : 0.0,
    str : ''
})


def option_find_type_default(
    type          : Type[T],
    type_present  : str,
    data          : ConfigBlock,
    option_name   : str,
    default_value : T
) -> T:
    if not option_name in data.params:
        return default_value
    else:
        param_data = data.params[option_name]

        try:
            return type(param_data.value)
        except:
            raise DarknetConfigLoadException(
                f'Param \'{param_data.name}\', '
                f'cannot convert value \'{param_data.value}\' to {type_present}',
                data.get_file_path(),
                param_data.line_number
            )

def option_find_type(
    type         : Type[T],
    type_present : str,
    data         : ConfigBlock, 
    option_name  : str
) -> T:
    default_values = cast(Dict[Type, Any], DEFAULT_VALUES)

    if not option_name in data.params:
        raise DarknetConfigLoadException(
            f'Problem in block \'{data.name}\', '
            f'not exist param \'{option_name}\'',
            data.get_file_path(),
            data.line_number
        )

    return option_find_type_default(
        type,
        type_present,
        data,
        option_name,
        default_values[type]
    )


def option_find_int_default(
    data          : ConfigBlock, 
    option_name   : str, 
    default_value : int
) -> int:
    return option_find_type_default(int, 'integer', data, option_name, default_value)


def option_find_int(data : ConfigBlock, option_name : str) -> int:
    return option_find_type(int, 'integer', data, option_name)


def option_find_float_default(
    data          : ConfigBlock, 
    option_name   : str, 
    default_value : float
) -> float:
    return option_find_type_default(float, 'float', data, option_name, default_value)


def option_find_float(data : ConfigBlock, option_name : str) -> float:
    return option_find_type(float, 'float', data, option_name)


def option_find_str_default(
    data          : ConfigBlock, 
    option_name   : str, 
    default_value : str
) -> str:
    return option_find_type_default(str, 'string', data, option_name, default_value)


def option_find_str(data : ConfigBlock, option_name : str) -> str:
    return option_find_type(str, 'string', data, option_name)


def option_find_type_list(type : Type[T], data : ConfigBlock, option_name : str) -> List[T]:
    if not option_name in data.params:
        raise DarknetConfigLoadException(
            f'Problem in block \'{data.name}\', ' 
            f'not exist param \'{option_name}\'',
            data.get_file_path(),
            data.line_number
        )

    param = data.params[option_name]

    splits = param.value.split(',')
    
    type_values : List[T] = list()

    for split_str in splits:
        try:
            int_val = type(split_str)
            type_values.append(int_val)
        except:
            raise DarknetConfigLoadException(
                f'Cannot convert string value \'{param.value}\' to List[{type}]',
                data.get_file_path(),
                param.line_number
            )

    return type_values


def option_find_int_list(data : ConfigBlock, option_name : str) -> List[int]:
    return option_find_type_list(int, data, option_name)


def option_find_float_list(data : ConfigBlock, option_name : str) -> List[float]:
    return option_find_type_list(float, data, option_name)


def option_validate_allow_parameters(data : ConfigBlock, allow_parameters : Set[str]):
    for key, param in data.params.items():
        if not param.name in allow_parameters:
            raise DarknetConfigLoadException(
                f'Not expected param \'{param.name}\' '
                f'in block \'{data.name}\'',
                data.get_file_path(),
                param.line_number
            )
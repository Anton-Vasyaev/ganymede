# 3rd party
from typing import Type, TypeVar, List, Any

T = TypeVar('T')

def check_cast(
    type       : Type[T], 
    ob         : Any, 
    param_name : str = ''
) -> T:
    if not isinstance(ob, type):
        data_message = 'Casting error. Invalid type for object'
        if param_name != '':
            data_message += f' \'{param_name}\''
        
        data_message += f'. Invalid type for object, expected \'{type}\', got \'{type(ob)}\'.'

        raise ValueError(data_message)

    return ob


def is_equal_types_b(objects : List[Any]) -> bool:
    if len(objects) < 1:
        raise ValueError(f'Invalid len of objects:{len(objects)}.')

    first_ob_type = objects[0]

    for ob in objects:
        if first_ob_type != type(ob):
            return False

    return True


def is_equal_types_val(objects : List[Any]) -> None:
    if not is_equal_types_b(objects):
        err_msg_list : List[str] = []

        err_msg_list.append('Objects is not of the same type: [')
        for ob in objects:
            err_msg_list.append(f'{type(ob)}.')
        err_msg_list.append('].')

        err_msg = ''.join(err_msg_list)

        raise ValueError(err_msg)
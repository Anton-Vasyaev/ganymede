# python
from numbers import Number


def __is_point_list(
    data,
):
    if not isinstance(data, list): return False
    
    if len(data) != 2: return False 
    
    return isinstance(data[0], Number) and isinstance(data[1], Number)


def __transform_list(
    points_data,
    transform_functor
):
    new_list = []
    for data in points_data:
        new_list.append(__transform_item(data, transform_functor))
        
    return new_list
    
    
def __transform_dict(
    points_data : dict,
    transform_functor
):
    transformed_dict = {}
    
    for key, value in points_data.items():
        transformed_dict[key] = __transform_item(value, transform_functor)
        
    return transformed_dict
            
            
def __transform_item(
    item,
    transform_functor
):
    transformed_data = None
    if isinstance(item, tuple):
        transformed_data = transform_functor(item)
    elif __is_point_list(item):
        transformed_data = transform_functor(item)
    elif isinstance(item, dict):
        transformed_data = __transform_dict(item, transform_functor)
    elif isinstance(item, list):
        transformed_data = __transform_list(item, transform_functor)
    else:
        raise ValueError(f'invalid type for perspective transform:{type(item)}')
        
    return transformed_data   


def delegate_transform_points(points_data, transform_functor):
    if points_data is None: return None
    return __transform_item(points_data, transform_functor)
# python
from typing import Callable, Any, cast
# project
from ganymede.math.primitives import Point2, is_point2


PointTransformer = Callable[[Point2], Point2]


def __is_sequence(data: Any) -> bool:
    return isinstance(data, list) or isinstance(data, tuple)


def __transform_list_or_tuple(
    points_data: Any,
    transform_functor: PointTransformer
) -> Any:
    new_list = []
    for data in points_data:
        new_list.append(__transform_item(data, transform_functor))

    return new_list


def __transform_dict(
    points_data: dict,
    transform_functor: PointTransformer
) -> dict:
    transformed_dict = {}

    for key, value in points_data.items():
        transformed_dict[key] = __transform_item(value, transform_functor)

    return transformed_dict


def __transform_item(
    item: Any,
    transform_functor: PointTransformer
) -> Any:
    transformed_data = None

    if is_point2(item):
        point = cast(Point2, item)
        point = transform_functor(point)
    elif isinstance(item, dict):
        transformed_data = __transform_dict(item, transform_functor)
    elif __is_sequence(item):
        transformed_data = __transform_list_or_tuple(item, transform_functor)
    else:
        raise ValueError(
            f'invalid type for perspective transform:{type(item)}')

    return transformed_data


def delegate_transform_points(points_data: Any, transform_functor: PointTransformer) -> Any:
    if points_data is None:
        return None
    return __transform_item(points_data, transform_functor)

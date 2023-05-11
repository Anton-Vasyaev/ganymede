# python
from dataclasses import dataclass
from typing      import List
# project
from ...option     import option_find_int_default, option_find_int_list, option_validate_allow_parameters
from ..config_info import ConfigBlock
from ...error      import DarknetConfigLoadException


@dataclass
class RouteLayer:
    layers : List[int]

    groups : int

    group_id : int

    config_block : ConfigBlock


ALLOW_ROUTE_PARAMS = set([
    'layers',
    'groups',
    'group_id'
])


def parse_route(data : ConfigBlock) -> RouteLayer:
    option_validate_allow_parameters(data, ALLOW_ROUTE_PARAMS)

    layers   = option_find_int_list(data, 'layers')
    groups   = option_find_int_default(data, 'groups', 1)
    group_id = option_find_int_default(data, 'group_id', 0)

    if group_id < 0 or group_id >= groups:
        raise DarknetConfigLoadException(
            f'Invalid group_id value:{group_id}, (groups:{groups})',
            data.get_file_path(),
            data.line_number
        )

    return RouteLayer(layers, groups, group_id, data)
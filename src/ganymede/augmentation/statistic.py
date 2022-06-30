# python
from copy import deepcopy
# project
import ganymede.random as g_random
from .parameters   import *
from .distribution import *


def create_random_padding(
    params,
    random_instance = None
):
    p  = params
    rs = random_instance

    left  = g_random.get_random_distance(p.left_pads[0], p.left_pads[1], rs)
    right = g_random.get_random_distance(p.right_pads[0], p.right_pads[1], rs)

    top    = g_random.get_random_distance(p.top_pads[0], p.top_pads[1], rs)
    bottom = g_random.get_random_distance(p.bottom_pads[0], p.top_pads[1], rs)

    return PaddingParameters(left, right, top, bottom)


def create_random_rotate2d(
    params,
    random_instance
):
    p  = params
    rs = random_instance

    angle = g_random.get_random_distance(p.min_angle, p.max_angle, rs)

    return Rotate2dParameters(angle)


def create_random_rotate3d(
    params,
    random_instance
):
    p  = params
    rs = random_instance

    x_angle = g_random.get_random_distance(p.x_angles[0], p.x_angles[1], rs)
    y_angle = g_random.get_random_distance(p.y_angles[0], p.y_angles[1], rs)
    z_angle = g_random.get_random_distance(p.z_angles[0], p.z_angles[1], rs)

    return Rotate3dParameters(x_angle, y_angle, z_angle)


def create_random_stretch(
    params,
    random_instance
):
    p  = params
    rs = random_instance

    offset = g_random.get_random_distance(p.min_offset, p.max_offset)

    s_type = p.type
    if s_type is None:
        s_type = g_random.get_random_enum(StretchType, rs)

    orientation = p.orientation
    if orientation is None:
        orientation = g_random.get_random_enum(StretchOrientation, rs)

    return StretchParameters(
        offset,
        s_type,
        orientation
    )


def create_random_basic_color(
    params,
    random_instance
):
    p  = params
    rs = random_instance

    red   = g_random.get_random_distance(p.red_values[0],   p.red_values[1],   rs)
    green = g_random.get_random_distance(p.green_values[0], p.green_values[1], rs)
    blue  = g_random.get_random_distance(p.blue_values[0],  p.blue_values[1],  rs)

    return BasicColorParameters(red, green, blue)


def create_random_mirror(
    params,
    random_instance = None
):
    p  = params
    rs = random_instance

    horizontal = g_random.get_random_bool(random_instance=rs) if p.horizontal else False
    vertical   = g_random.get_random_bool(random_instance=rs) if p.vertical   else False

    return MirrorParameters(horizontal, vertical)


def create_random_augmentation_selector(
    dist,
    random_instance = None
):
    t = type(dist)

    if t is BasicColorDistribution:
        return create_random_basic_color(dist, random_instance)
    elif t is MirrorDistribution:
        return create_random_mirror(dist, random_instance)
    elif t is PaddingDistribution:
        return create_random_padding(dist, random_instance)
    elif t is Rotate2dDistribution:
        return create_random_rotate2d(dist, random_instance)
    elif t is Rotate3dDistribution:
        return create_random_rotate3d(dist, random_instance)
    elif t is StretchDistribution:
        return create_random_stretch(dist, random_instance)
    else:
        raise NotImplementedError(f'not implemented create_random_augmentation_selector for {t}')


def create_random_aug_stages(
    distribution_parameters,
    random_instance = None
):
    ds = distribution_parameters
    rs = random_instance

    stages = []

    if not ds.mirror is None:
        if ds.mirror.horizontal or ds.mirror.vertical: 
            stages.append(create_random_mirror(ds.mirror, rs))
    
    if not ds.padding is None: stages.append(create_random_padding(ds.padding, rs))


    middle_stage_types = []
    if not ds.rotate2d is None: middle_stage_types.append(AugmentationType.ROTATE_2D)
    if not ds.rotate3d is None: middle_stage_types.append(AugmentationType.ROTATE_3D)
    if not ds.stretch  is None: middle_stage_types.append(AugmentationType.STRETCH)

    middle_stage = None
    if len(middle_stage_types) != 0:
        random_type = g_random.choice(middle_stage_types, rs)
        if random_type == AugmentationType.ROTATE_2D:
            middle_stage = create_random_rotate2d(ds.rotate2d, rs)
        elif random_type == AugmentationType.ROTATE_3D:
            middle_stage = create_random_rotate3d(ds.rotate3d, rs)
        elif random_type == AugmentationType.STRETCH:
            middle_stage = create_random_stretch(ds.stretch, rs)
        
    if not middle_stage is None: stages.append(middle_stage)

    if not ds.basic_color is None: stages.append(create_random_basic_color(ds.basic_color, rs))

    return stages


def make_augmentation_distribution(
    data_list,
    distribution_parameters,
    aug_size,
    random_instance = None
):
    data_len = len(data_list)

    new_data_list = []

    append_length = int(len(data_list) * aug_size)

    start_idx = 0
    while start_idx < append_length:
        aug_length = min(append_length - start_idx, data_len)

        copy_data = deepcopy(g_random.sample(data_list, aug_length, random_instance))

        for idx in range(len(copy_data)):
            copy_data[idx].meta_info['aug_stages'] = create_random_aug_stages(
                distribution_parameters,
                random_instance
            )

        new_data_list += copy_data

        start_idx += aug_length

    new_data_list += deepcopy(data_list)

    return new_data_list
# python
from dataclasses import dataclass
from typing import Tuple
# 3rd party
import cv2 as cv
import numpy as np
# project
import ganymede.numpy.validation as np_valid
import ganymede.math.bbox2 as m_bbox2
from ganymede.math.primitives import BBox2, Size2

from ganymede.imaging.processing import cast_one_channel_img, create_view_with_channel
from ganymede.imaging.auxiliary import get_channels_of_numpy

def check_empty_int_box(
    box : Tuple[int, int, int, int]
):
    return box[1] - box[0] == 0 or box[3] - box[2] == 0


def __apply_patch_on_uint8(
    canvas_patch : np.ndarray,
    draw_patch   : np.ndarray,
    mask_patch   : np.ndarray
):
    canvas_patch_f32 = canvas_patch.astype(np.float32) / 255.0
    draw_patch_f32   = draw_patch.astype(np.float32) / 255.0
    mask_patch_f32   = mask_patch.astype(np.float32) / 255.0

    result = canvas_patch_f32 * (-mask_patch_f32 + 1.0) + draw_patch_f32 * mask_patch_f32
    result *= 255.0

    canvas_patch[...] = result.astype(np.uint8)


def __apply_patch_on_normalized_f32(
    canvas_patch : np.ndarray,
    draw_patch   : np.ndarray,
    mask_patch   : np.ndarray
):
    canvas_patch[...] = canvas_patch[...] * (-mask_patch + 1.0) + (draw_patch * mask_patch)


@dataclass
class _PlaceInfo:
    place_resize : Tuple[int, int]
    
    place_coords : Tuple[int, int, int, int]

    src_offsets : Tuple[int, int, int, int]


def __calculate_place_coords_for_src_dst(
    dst_size : Tuple[int, int],
    draw_box : BBox2
):
    dst_w, dst_h = dst_size

    draw_x1, draw_y1, draw_x2, draw_y2 = draw_box

    abs_place_x1 = int(draw_x1 * dst_w)
    abs_place_y1 = int(draw_y1 * dst_h)

    abs_place_x2 = int(draw_x2 * dst_w)
    abs_place_y2 = int(draw_y2 * dst_h)

    abs_place_w = abs_place_x2 - abs_place_x1
    abs_place_h = abs_place_y2 - abs_place_y1

    src_offset_x1 = max(0, -abs_place_x1)
    src_offset_x2 = min(abs_place_w, abs_place_w + (dst_w - abs_place_x2))

    src_offset_y1 = max(0, -abs_place_y1)
    src_offset_y2 = min(abs_place_h, abs_place_h + (dst_h - abs_place_y2))

    dst_x1 = max(0, abs_place_x1)
    dst_x2 = min(dst_w, abs_place_x2)

    dst_y1 = max(0, abs_place_y1)
    dst_y2 = min(dst_h, abs_place_y2)

    return _PlaceInfo(
        place_resize=(abs_place_w, abs_place_h),
        place_coords=(dst_x1, dst_y1, dst_x2, dst_y2),
        src_offsets=(src_offset_x1, src_offset_y1, src_offset_x2, src_offset_y2)
    )


def place_image_on_image(
    dst_img     : np.ndarray,
    src_img     : np.ndarray,
    draw_box    : BBox2,
    resize_flag : int = cv.INTER_AREA
):
    dst_channels = get_channels_of_numpy(dst_img)
    src_channels   = get_channels_of_numpy(src_img)

    assert dst_channels == src_channels
    assert dst_img.dtype == src_img.dtype

    dst_h, dst_w = dst_img.shape[0:2]
    inf = __calculate_place_coords_for_src_dst(
        (dst_w, dst_h),
        draw_box
    )

    src_img = cv.resize(src_img, inf.place_resize, interpolation=resize_flag)
    src_img = create_view_with_channel(src_img)

    place_x1, place_y1, place_x2, place_y2 = inf.place_coords

    src_x1, src_y1, src_x2, src_y2 = inf.src_offsets

    dst_img[place_y1:place_y2, place_x1:place_x2] = src_img[
        src_y1:src_y2,
        src_x1:src_x2
    ]
    

def place_mask_on_mask(
    dst_img     : np.ndarray,
    src_img     : np.ndarray,
    draw_box    : BBox2,
    resize_flag : int = cv.INTER_AREA
):
    dst_channels = get_channels_of_numpy(dst_img)
    src_channels   = get_channels_of_numpy(src_img)

    assert dst_channels == src_channels
    assert dst_img.dtype == src_img.dtype

    dst_h, dst_w = dst_img.shape[0:2]
    inf = __calculate_place_coords_for_src_dst(
        (dst_w, dst_h),
        draw_box
    )

    src_img = cv.resize(src_img, inf.place_resize, interpolation=resize_flag)
    src_img = create_view_with_channel(src_img)

    place_x1, place_y1, place_x2, place_y2 = inf.place_coords

    src_x1, src_y1, src_x2, src_y2 = inf.src_offsets

    dst = dst_img[place_y1:place_y2, place_x1:place_x2]
    src = src_img[src_y1:src_y2, src_x1:src_x2]
    
    dst[...] = np.maximum(src, dst)


def draw_image_on_image(
    canvas_img  : np.ndarray,
    drawed_img  : np.ndarray,
    drawed_mask : np.ndarray,
    draw_box    : BBox2,
    resize_flag : int = cv.INTER_AREA,
    normalized  : bool = True
) -> None:
    assert drawed_img.shape[0:2] == drawed_mask.shape[0:2]
    canvas_channels = get_channels_of_numpy(canvas_img)
    draw_channels   = get_channels_of_numpy(drawed_img)

    assert canvas_channels == draw_channels
    assert canvas_img.dtype == drawed_img.dtype
    assert drawed_img.dtype == drawed_mask.dtype

    canvas_h, canvas_w = canvas_img.shape[0:2]
    inf = __calculate_place_coords_for_src_dst(
        (canvas_w, canvas_h),
        draw_box
    )

    drawed_img  = cv.resize(drawed_img,  inf.place_resize, interpolation=resize_flag)
    drawed_mask = cv.resize(drawed_mask, inf.place_resize, interpolation=resize_flag)

    canvas_img  = create_view_with_channel(canvas_img)
    drawed_img  = create_view_with_channel(drawed_img)
    drawed_mask = create_view_with_channel(drawed_mask)

    place_x1, place_y1, place_x2, place_y2 = inf.place_coords
    canvas_patch = canvas_img[place_y1:place_y2, place_x1:place_x2]

    src_x1, src_y1, src_x2, src_y2 = inf.src_offsets
    draw_patch = drawed_img[
        src_y1:src_y2,
        src_x1:src_x2
    ]
    mask_patch = drawed_mask[
        src_y1:src_y2,
        src_x1:src_x2
    ]
    mask_patch = cast_one_channel_img(mask_patch, canvas_channels)

    if canvas_img.dtype == np.float32 and normalized:
        __apply_patch_on_normalized_f32(
            canvas_patch,
            draw_patch,
            mask_patch
        )
    elif canvas_img.dtype == np.uint8:
        __apply_patch_on_uint8(canvas_patch, draw_patch, mask_patch)
    elif canvas_img.dtype == np.float32 and normalized:
        raise Exception(f'Not implemented for float32 not normalized image (from 0 to 255 range).')
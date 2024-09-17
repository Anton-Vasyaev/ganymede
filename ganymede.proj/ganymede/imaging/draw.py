# python
from typing import Tuple
# 3rd party
import opencv as cv
import numpy as np
# project
import ganymede.math.bbox2 as m_bbox2
from ganymede.math.primitives import BBox2


def check_empty_int_box(
    box : Tuple[int, int, int, int]
):
    return box[1] - box[0] == 0 or box[3] - box[2] == 0

# TODO
# finish this method late
def draw_image_on_image(
    src_img : np.ndarray,
    dst_img : np.ndarray,
    draw_box : BBox2,
    resize_flag : int = cv.INTER_AREA
) -> None:
    src_h, src_w = src_img.shape[0:2]
    dst_h, dst_w = dst_img.shape[0:2]


    draw_x1, draw_y1, draw_x2, draw_y2 = draw_box

    draw_w = m_bbox2.width(draw_box)
    draw_h = m_bbox2.height(draw_box)

    src_x1 = max(0.0, draw_x1)
    src_y1 = max(0.0, draw_y1)
    src_x2 = min(1.0, draw_x2)
    src_y2 = min(1.0, draw_y2)

    dst_x1 = (src_x1 - draw_x1) / draw_w
    dst_y1 = (src_y1 - draw_y1) / draw_h
    dst_x2 = (src_x2 - draw_x1) / draw_w
    dst_y2 = (src_y2 - draw_y2) / draw_h

    draw_w = m_bbox2.width(draw_box)
    draw_h = m_bbox2.height(draw_box)

    abs_draw_w = int(draw_w * src_w)
    abs_draw_h = int(draw_h * src_h)

    abs_src_x1, abs_src_y1 = int(src_x1 * src_w), int(src_y1 * src_h)
    abs_src_x2, abs_src_y2 = int(src_x2 * src_w), int(src_y2 * src_h)
    abs_src_box            = (abs_src_x1, abs_src_y1, abs_src_x2, abs_src_y2)

    abs_dst_x1, abs_dst_y1 = int(dst_x1 * dst_w), int(dst_y1 * dst_h)
    abs_dst_x2, abs_dst_y2 = int(dst_x2 * dst_w), int(dst_y2 * dst_h)
    abs_dst_box            = (abs_dst_x1, abs_dst_y1, abs_dst_x2, abs_dst_y2)

    if check_empty_int_box(abs_src_box) or check_empty_int_box(abs_dst_box):
        return
    
    resize_w = abs_src_x2 - abs_src_x1
    resize_h = abs_src_y2 - abs_src_y1

    resize_draw_box = cv.resize(draw_box, (resize_w, resize_h))
    
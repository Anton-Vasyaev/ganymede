# python
from typing import Tuple, List, Optional
# 3rd party
import numpy as np
# project
import ganymede.opencv as g_cv

from .data import DebugImageInfo

def show_debug_img_info(
    info         : DebugImageInfo,
    show_size    : Optional[Tuple[int, int]] = None,
    wait_ms      : int = 0,
    escape_catch : bool = True
) -> g_cv.OpenCVWindowKey:
    
    show_w, show_h = 1600, 900

    if not show_size is None:
        show_w, show_h = show_size

    debug_frames : List[Tuple[str, np.ndarray]] = list()

    for draw_id, draw in info.draws.items():
        image_id = draw.image_id

        show_frame = np.array([])
        if image_id is None:
            show_frame = np.zeros((show_h, show_w, 3), dtype=np.uint8)
        else:
            show_frame = info.images[image_id]
            if not show_size is None:
                show_frame = g_cv.resize_frame(img, (show_w, show_h))
            else:
                show_frame = show_frame.copy()

        g_cv.draw_canvas(show_frame, draw.canvas)

        debug_frames.append((draw_id, show_frame))


    for img_id, img in info.images.items():
        img = g_cv.resize_frame(img, (show_w, show_h))

        debug_frames.append((img_id, img))

    return g_cv.imshow_multi(debug_frames, wait_ms, escape_catch=escape_catch)
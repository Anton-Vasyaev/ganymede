# python
from copy import deepcopy
# 3rd party
import numpy as np
from   PIL   import ImageFont, ImageDraw, Image


GRAY_RED   = 0.299
GRAY_GREEN = 0.587
GRAY_BLUE  = 0.114


def draw_text_list(
    img,
    text_list,
    font_size=0.05
):
    is_gray_img_flag = False

    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img.shape = img.shape[:2]
            is_gray_img_flag = True
    elif len(img.shape) == 2:
        is_gray_img_flag = True

    img_h, img_w = img.shape[0:2]

    font_size = int(font_size * img_h)

    font = ImageFont.truetype("arial.ttf", font_size)

    if np.issubdtype(img.dtype, np.floating):
        copy_img = (img * 255).astype(np.uint8)
    else:
        copy_img = img

    img_pil = Image.fromarray(copy_img)
    draw    = ImageDraw.Draw(img_pil)

    for text_block in text_list:
        text, left_corner, color = text_block

        x, y    = left_corner
        x, y    = int(x * img_w), int(y * img_h)
        r, g, b = color

        real_color = (b, g, r)
        if is_gray_img_flag:
            real_color = int(r * GRAY_RED + g * GRAY_GREEN + b * GRAY_BLUE)

        draw.text((x, y), text, font=font, fill=real_color)

    draw_img = np.array(img_pil)

    img[:] = draw_img
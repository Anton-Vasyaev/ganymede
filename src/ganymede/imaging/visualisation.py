# python
from typing import List, Tuple
# 3rd party
import numpy as np
import cv2   as cv
# project
import ganymede.opencv as g_cv
from .auxiliary import get_channels
from .processing import cast_one_channel_img



def form_images_tile(
    images       : List[np.ndarray], 
    resize       : Tuple[int, int] = None, 
    frame_size   : Tuple[int, int] = True,
    resize_inter : int             = cv.INTER_AREA
) -> np.ndarray:
    im_len = len(images)

    # вычисляем максимальную высоту и ширину изображения
    max_h, max_w = 0, 0
    for img in images:
        h, w = img.shape[0:2]
        max_h, max_w = max(max_h, h), max(max_w, w)

    # вычисляем максимальную длину таблицы (высота, ширина)
    max_tile_dim_size = 0
    idx = 1
    while True:
        if im_len < idx * idx:
            max_tile_dim_size = idx
            break
        idx += 1

    # вычисляем количество строк и столбцов
    tile_w = max_tile_dim_size
    tile_h = im_len // max_tile_dim_size
    mod    = im_len % max_tile_dim_size
    if mod != 0: tile_h += 1

    # Формируем таблицу изображений 
    # Те ячейки которые оказались пустыми, заполняются
    # Пустым изображением
    placeholder_cell = None
    if mod != 0: placeholder_cell = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    table_list = []
    for idx in range(tile_h):
        table_list.append([placeholder_cell] * tile_w)

    # Заполняем таблицу преобразованными изображениями
    row_idx = 0
    column_idx = 0
    for img in images:
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).astype(np.uint8)

        if get_channels(img) == 1: img = cast_one_channel_img(img)
        img = g_cv.resize_frame(img, (max_w, max_h))
        img_h, img_w = img.shape[0:2]

        fill_img = np.zeros((max_h, max_w, 3), dtype=np.uint8)

        fill_x = (max_w - img_w) // 2
        fill_y = (max_h - img_h) // 2

        fill_img[fill_y:fill_y+img_h, fill_x:fill_x+img_w] = img

        table_list[row_idx][column_idx] = fill_img

        column_idx += 1
        if column_idx == tile_w:
            column_idx = column_idx % tile_w
            row_idx += 1

    # Производим конкатенацию изображений по строкам
    concat_rows = []
    for table_row in table_list:
        concat_rows.append(np.concatenate(table_row, axis=1))

    concat_table = np.concatenate(concat_rows, axis=0)

    if not resize is None:
        if frame_size:
            concat_table = g_cv.resize_frame(concat_table, resize, resize_inter)
        else:
            concat_table = g_cv.resize_frame(concat_table, resize, resize_inter)

    return concat_table

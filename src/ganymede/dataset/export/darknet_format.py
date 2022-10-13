# python
from typing import List
# 3rd party
import numpy as np
from pathlib import Path
# project
import ganymede.opencv as g_cv
from ganymede.dataset.data.bbox_object_markup import BBoxObjectMarkup


def write_class_names(obj_file_path, class_names):
    with open(obj_file_path, 'w') as fh:
        fh.writelines(class_names)


def convert_to_yolo_objects(objects : List[BBoxObjectMarkup]):
    yolo_objects = []
    for object in objects:
        bbox     = object.bbox
        class_id = object.class_id

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0.0, 1.0)

        w, h = x2 - x1, y2 - y1

        x_c, y_c = x1 + w / 2, y1 + h / 2

        yolo_objects.append([class_id, x_c, y_c, w, h])

    return yolo_objects


def export_part(
    list_file_handler,
    dataset_loader,
    dir_path,
    start_idx = None
):
    dir_p = Path(dir_path)

    img_idx = start_idx if not start_idx is None else 0

    for idx in range(len(dataset_loader)):
        print(f'export to dir:{dir_path}:[{idx+1}/{len(dataset_loader)}]')
        img, objects = dataset_loader[idx]

        yolo_objects = convert_to_yolo_objects(objects)

        img_path = dir_p / f'{img_idx}.png'
        txt_path = dir_p / f'{img_idx}.txt'

        g_cv.imwrite(img, str(img_path))

        with open(str(txt_path), 'w') as fh:
            for object in yolo_objects:
                ff = '{:.6f}'
                class_id, x_c, y_c, w, h = object
                x_c, y_c, w, h = ff.format(x_c), ff.format(y_c), ff.format(w), ff.format(h)
                fh.write(f'{class_id} {x_c} {y_c} {w} {h}\n')
        
        list_file_handler.write(f'{img_path}\n')

        img_idx += 1

    return img_idx


def export_to_darknet_format(
    export_dir,
    train_dataset_loader,
    test_dataset_loader
):
    output_path = Path(export_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_path = output_path / 'images'
    images_path.mkdir(parents=True, exist_ok=True)

    backup_path = output_path / 'backup'
    backup_path.mkdir(parents=True, exist_ok=True)

    names_path = output_path / 'obj.names'

    train_list_path = output_path / 'train.txt'
    test_list_path  = output_path / 'test.txt' 

    class_names = train_dataset_loader.get_class_names()

    # Пишем файл с именами классов
    write_class_names(str(output_path / 'obj.names'), class_names)

    # Записываем файл с конфигурацией датасеты
    with open(str(output_path / 'obj.data'), 'w') as fh:
        fh.write(
            f'classes = {len(class_names)}\n'
            f'train = {train_list_path}\n'
            f'valid = {test_list_path}\n'
            f'names = {names_path}\n'
            f'backup = {backup_path}'
        )

    # Записываем списки с изображениями в тренировочном и тестовом датасете
    with open(train_list_path, 'w') as list_handler:
        train_imgs_path = images_path / 'train'
        export_part(list_handler, train_dataset_loader, str(train_imgs_path))

    with open(test_list_path, 'w') as list_handler:
        test_imgs_path = images_path / 'test'
        export_part(list_handler, test_dataset_loader, str(test_imgs_path))
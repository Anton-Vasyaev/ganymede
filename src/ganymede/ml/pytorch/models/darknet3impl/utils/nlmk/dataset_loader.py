# python
import random
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ElementTree

from enum import IntEnum

# 3rd party
import numpy as np
import cv2 as cv
import torch

from pathlib import Path
# project
from utils.augmentations import horisontal_flip


class DetectionType(IntEnum):
    LADLE = 0
    TRUCK = 1
    SLAG_TRUCK = 2


def get_detection_type_of_label(label):
    if label == 'Ladle':
        return DetectionType.LADLE
    elif label == 'Truck':
        return DetectionType.TRUCK
    elif label == 'Slag-Truck':
        return DetectionType.SLAG_TRUCK
    else:
        raise f'not available type:{label}'


def read_xml_pathes(xml_path):

    image_infos = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_path = Path(xml_path)

    images = root.findall('image')
    for image in images:
        image_info = {}
        image_info['boxes'] = []
        image_info['id'] = image.get('id')
        image_info['file_path'] = image.get('name')
        boxes = image.findall('box')
        if len(boxes) == 0:
            continue

        img_path = Path(image.get('name'))

        if img_path.is_absolute():
            image_info['file_path'] = str(img_path)
        else:
            image_info['file_path'] = str(xml_path.parent / img_path)

        for box in boxes:
            label = None
            try:
                label = get_detection_type_of_label(box.get('label'))
            except:
                continue

            xtl     = float(box.get('xtl'))
            ytl     = float(box.get('ytl'))
            xbr     = float(box.get('xbr'))
            ybr     = float(box.get('ybr'))

            image_info['boxes'].append(
                {
                    'label' : label,
                    'xtl' : xtl,
                    'ytl' : ytl,
                    'xbr' : xbr,
                    'ybr' : ybr,
                    'parents' : []
                }
            )
        image_infos.append(image_info)

    return image_infos
        


class NLMKConfigLoader:
    @staticmethod
    def get_labels():
        return [
            'Ladle',
            'Truck',
            'Slag-Truck'
        ]

    def __init__(self, xml_pathes):
        self.datas = []

        for xml_path in xml_pathes:
            self.datas += (read_xml_pathes(xml_path))

    
    def __len__(self):
        return len(self.datas)

    
    def __getitem__(self, idx):
        return self.datas[idx]



class NLMKDetectDatasetIterator:
    def __init__(self, loader):
        self._loader = loader

        self.idx = 0
        self.end = len(self._loader)

    def __next__(self):
        if self.idx >= self.end:
            raise StopIteration
        else:
            data = self._loader[self.idx]
            self.idx += 1
            return data
    


class NLMKDetectDatasetLoader:
    def __init__(self, config_loader, resize=(412, 412), batch_size=4, shuffle=True):
        self.datas  = config_loader.datas
        self.resize = resize
        random.shuffle(self.datas)
        self.batch_size = batch_size


    def __len__(self):
        return len(self.datas) // self.batch_size


    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError
        offset = idx * self.batch_size

        image_infos = self.datas[offset:offset+self.batch_size]
        images  = []
        targets = []

        result_path = []

        img_idx = 0
        for image_info in image_infos:
            result_path.append(image_info['file_path'])
            img = cv.imread(image_info['file_path'], cv.IMREAD_COLOR)
            (img_h, img_w, _) = img.shape
            img = cv.resize(img, (412, 412), cv.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img.shape = (1,) + img.shape
            images.append(img)

            boxes = image_info['boxes']
            for box in boxes:
                float_label = float(box['label'])
                x1 = box['xtl'] / img_w
                y1 = box['ytl'] / img_h
                x2 = box['xbr'] / img_w
                y2 = box['ybr'] / img_h

                w = x2 - x1
                h = y2 - y1
                x_center = x1 + w / 2
                y_center = y1 + h / 2

                targets.append(np.array([img_idx, float_label, x_center, y_center, w, h]))

            img_idx += 1

        images  = np.vstack(images).transpose(0, 3, 1, 2)
        targets = np.vstack(targets)

        images  = torch.from_numpy(images)
        targets = torch.from_numpy(targets)

        if idx % 2 == 0:
            images, targets = horisontal_flip(images, targets)

        return image_infos, images.type(torch.float32), targets.type(torch.float32)

    
    def __iter__(self):
        return NLMKDetectDatasetIterator(self)



if __name__ == '__main__':
    pass
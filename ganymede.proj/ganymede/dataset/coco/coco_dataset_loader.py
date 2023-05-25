# python
import os.path as path
from typing      import Tuple, Optional, List, Dict, Any, cast
from dataclasses import dataclass
from copy        import deepcopy
# 3rd party
import numpy as np
from autofast.config import *
# project
import ganymede.opencv as g_cv
import ganymede.json   as g_js

import ganymede.math.point2 as m_p2
import ganymede.math.bbox2  as m_bbox2

from ganymede.math.primitives import Polygon2, BBox2

from ..data import SegmentObjectMarkup


@dataclass
class Category:
    id : int

    name : str

    supercategory : str


@dataclass
class Image:
    id : int

    license : int

    file_name : str

    height : int

    width : int


SEGMENT_LIST_FAIL_MESSAGE = f'fail to encode segmentation list from coco.'


def decode_coco_segmentation_list(node : Node) -> List[Polygon2]:
    if not isinstance(node, ListNode):
        raise Exception(
            f'failed to decode list of segments. expected {ListNode}, gotted:{type(node)}.'
        )

    list_node = cast(ListNode, node)

    segments_list : List[Polygon2] = list()

    valid_point_list_exc = Exception(
        f'failed to decode segment polygon. Segment polygon must be list of numbers whose length '
        f'is multiply by 2.'
    )
    for segment_node in list_node.list_data:
        if not isinstance(segment_node, ListNode):
            raise Exception(
                f'Failed to decode segmentation polygon. expected {ListNode}, gotted:{type(segment_node)}.'
            )
        
        segment_list_node = cast(ListNode, segment_node)
    
        segment_values = list()
        for elem_node in segment_list_node.list_data:
            if not isinstance(elem_node, ValueNode):
                raise valid_point_list_exc
            
            elem_value_node = cast(ValueNode, elem_node)
            elem_value      = elem_value_node.value
            segment_values.append(elem_value)    


        segment_values_len = len(segment_values)
        if segment_values_len % 2 != 0:
            raise valid_point_list_exc
        
        segment_poly = list()
        for idx in range(len(segment_values) // 2):
            x = segment_values[2 * idx]
            y = segment_values[2 * idx + 1]
            segment_poly.append((x, y))

        segments_list.append(segment_poly)

    return segments_list


@dataclass
class Annotation:
    segmentation : List[Polygon2] = field_meta(decoder=decode_coco_segmentation_list)

    area : float

    iscrowd : int

    image_id : int

    bbox : BBox2

    category_id : int

    id : int


@dataclass
class CocoJsonData:
    image : Dict[str, Any]

    annotations : List[Dict[str, Any]]


@dataclass
class CocoExampleInfo:
    image : Image
    


class CocoDatasetLoader:
    ''' 
    Provides loading for COCO dataset(Common Objects in Context).
    See information about COCO: https://cocodataset.org/#home
    '''
    
    __images_dir : str
    
    __allowed_classes : Optional[List[str]]

    __categories : Dict[int, Any]

    __examples : List[CocoJsonData]


    def filter_examples(self, data_list : List[CocoJsonData]) -> List[CocoJsonData]:
        filtered_data_list = list()

        for data in data_list:
            anno_list = data.annotations
            filtered_anno_list = list()

            validate_flag = True

            for anno in anno_list:
                category_id = anno['category_id']
                category_name = self.__categories[category_id]['name']

                segmentation_list = anno['segmentation']
                if not isinstance(segmentation_list, list):
                    validate_flag = False
                    break

                if category_name in self.__allowed_classes or self.__allowed_classes == None:
                    filtered_anno_list.append(anno)

            if validate_flag and len(filtered_anno_list) != 0:
                copy_data = deepcopy(data)
                copy_data.annotations = filtered_anno_list
                filtered_data_list.append(copy_data)

        return filtered_data_list


    def __init__(
        self,
        markup_path     : str,
        images_dir      : str,
        allowed_classes : Optional[List[str]] = None
    ):
        self.__images_dir      = images_dir
        self.__allowed_classes = allowed_classes

        categories_dict : Dict[int, Any] = dict()
        images_dict     : Dict[int, Any] = dict()

        annotations_dict : Dict[int, List[Dict[str, Any]]] = dict()
        
        data_json = g_js.load_from_file(markup_path)

        # loading categories
        categories = data_json['categories']
        for idx in range(len(categories)):
            category = categories[idx]
            categories_dict[category['id']] = category

            images = data_json['images']
        self.__categories = categories_dict

        # loading images
        for idx in range(len(images)):
            img                 = images[idx]
            img_id              = img['id']
            images_dict[img_id] = img

        # loading annotations
        annotations = data_json['annotations']
        for idx in range(len(annotations)):
            anno = annotations[idx]

            image_id = anno['image_id']

            if not image_id in annotations_dict:
                annotations_dict[image_id] = list()

            annotations_dict[image_id].append(anno)

        # form examples
        examples : List[CocoJsonData] = list()
        for image_id, annotation in annotations_dict.items():
            image       = images_dict[image_id]
            
            examples.append(CocoJsonData(image, annotation))

        self.__examples = self.filter_examples(examples)


    def __len__(self) -> int:
        return len(self.__examples)
    

    def __getitem__(self, idx : int) -> Tuple[np.ndarray, List[SegmentObjectMarkup]]:
        example = self.__examples[idx]

        image_data_json  = example.image
        annotations_json = example.annotations

        image_data = deserialize_config(Image, image_data_json)
        img_box    = (0.0, 0.0, image_data.width, image_data.height)

        image_path = path.join(self.__images_dir, image_data.file_name)
        img        = g_cv.imread(image_path)

        annotations_list : List[Annotation] = list()
        for anno_json in annotations_json:
            anno = deserialize_config(Annotation, anno_json)

            annotations_list.append(anno)

        for anno in annotations_list:
            for idx in range(len(anno.segmentation)):
                segment = anno.segmentation[idx]
                segment = [m_p2.normalize_on_contour(p, img_box) for p in segment]
                anno.segmentation[idx] = segment

        objects = list()
        for anno in annotations_list:
            objects.append(
                SegmentObjectMarkup(
                    anno.segmentation,
                    anno.category_id,
                )
            )

        return img, objects
    

    def get_class_names(self) -> Dict[int, str]:
        class_names = dict()
        for category_id, category in self.__categories.items():
            name = category['name']
            class_names[category_id] = name

        return class_names
    

    def get_example_info(self, idx : int) -> CocoExampleInfo:
        example = self.__examples[idx]

        image = deserialize_config(Image, example.image)

        return CocoExampleInfo(image)
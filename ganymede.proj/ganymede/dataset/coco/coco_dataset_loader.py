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

from ganymede.math.primitives import Polygon2, BBox2

from ..i_dataset_loader import IDatasetLoader

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
                f'Failed to decode segmentation polygon. '
                f'expected {ListNode}, gotted:{type(segment_node)}.'
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
    


class CocoDatasetLoader(IDatasetLoader[np.ndarray, List[SegmentObjectMarkup]]):
    ''' 
    Provides loading for COCO dataset(Common Objects in Context).
    See information about COCO: https://cocodataset.org/#home
    '''

#region data

    __images_dir : str
    
    __allowed_classes : List[str]

    __enable_empty_markup : bool

    __categories : Dict[int, Any]

    __examples : List[CocoJsonData]

    __image_id_access : Dict[int, int]

#endregion

#region private_methods 

    def __filter_examples(self, data_list : List[CocoJsonData]) -> List[CocoJsonData]:
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
                
                if category_name in self.__allowed_classes:
                    filtered_anno_list.append(anno)

            if validate_flag and (len(filtered_anno_list) != 0 or self.__enable_empty_markup):
                copy_data = deepcopy(data)
                copy_data.annotations = filtered_anno_list
                filtered_data_list.append(copy_data)

        return filtered_data_list

#endregion

#region construct_and_destruct

    def __init__(
        self,
        markup_path         : str,
        images_dir          : str,
        allowed_classes     : Optional[List[str]] = None,
        enable_empty_markup : bool                = False 
    ):
        self.__images_dir = images_dir

        self.__enable_empty_markup = enable_empty_markup

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

        if allowed_classes is None:
            allowed_classes = list()
            for category_id, category in self.__categories.items():
                category_name = category['name']
                allowed_classes.append(category_name)
        self.__allowed_classes = allowed_classes

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

        self.__examples = self.__filter_examples(examples)
        self.__examples.sort(key = lambda e : e.image['id'])

        # form image id access
        self.__image_id_access = dict()
        for example_id in range(len(self.__examples)):
            example = self.__examples[example_id]
            image   = deserialize_config(Image, example.image)

            self.__image_id_access[image.id] = example_id

#endregion

#region IDatasetLoader implementation

    def __len__(self) -> int:
        return len(self.__examples)


    def __getitem__(self, idx : int) -> Tuple[np.ndarray, List[SegmentObjectMarkup]]:
        img = self.get_image(idx)

        objects = self.get_markup(idx)

        return img, objects

#endregion
 
#region methods

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
    

    def get_markup(self, idx : int) -> List[SegmentObjectMarkup]:
        example = self.__examples[idx]

        image_data_json  = example.image
        annotations_json = example.annotations

        image_data = deserialize_config(Image, image_data_json)
        img_box    = (0.0, 0.0, image_data.width, image_data.height)

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

        return objects
    

    def get_image(self, idx : int) -> np.ndarray:
        example = self.__examples[idx]

        image_data_json  = example.image

        image_data = deserialize_config(Image, image_data_json)

        image_path = path.join(self.__images_dir, image_data.file_name)
        img        = g_cv.imread(image_path)

        return img

    def get_example_idx_by_image_id(self, image_id : int) -> int:
        return self.__image_id_access[image_id]

#endregion
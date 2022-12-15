'''
# python
import math
# 3rd party
import numpy as np
import cv2 as cv
# project
from ..auxiliary import delegate_aug_rectangles_from_points

DEGREE_PER_RADIAN = 180 / math.pi

class Augmentation3d:
    @staticmethod
    def get_perspective_matrix(angles, img_size):
        w, h = img_size

        rotx, roty, rotz = angles # set these first
        rotx = -rotx / DEGREE_PER_RADIAN
        roty = -roty / DEGREE_PER_RADIAN
        rotz = -rotz / DEGREE_PER_RADIAN

        f = 2; # this is also configurable, f=2 should be about 50mm focal length

        cx, sx = math.cos(rotx), math.sin(rotx)
        cy, sy = math.cos(roty), math.sin(roty)
        cz, sz = math.cos(rotz), math.sin(rotz)

        roto = np.float32(
            [
                [cz * cy, cz * sy * sx - sz * cx],
                [sz * cy, sz * sy * sx + cz * cx],
                [-sy, cy * sx]
            ]
        )

        pt = np.float32(
            [
                [ -w / 2, -h / 2 ], 
                [  w / 2, -h / 2 ], 
                [  w / 2,  h / 2 ], 
                [ -w / 2,  h / 2 ]
            ]
        )

        ptt = np.zeros((4, 2), dtype=np.float32)

        for i in range(4):
            pz = pt[i,0] * roto[2, 0] + pt[i, 1] * roto[2, 1]
            ptt[i, 0] = w / 2 + (pt[i, 0] * roto[0, 0] + pt[i, 1] * roto[0, 1]) * f * h / (f * h + pz)
            ptt[i, 1] = h / 2 + (pt[i, 0] * roto[1, 0] + pt[i, 1] * roto[1, 1]) * f * h / (f * h + pz)

        in_pt = np.float32(
            [
                [0, 0], [w, 0], [w, h], [0, h]
            ]
        )

        out_pt = np.float32(
            [
                [ptt[0, 0], ptt[0, 1]],
                [ptt[1, 0], ptt[1, 1]],
                [ptt[2, 0], ptt[2, 1]],
                [ptt[3, 0], ptt[3, 1]]
            ]
        )

        transform = cv.getPerspectiveTransform(in_pt, out_pt)

        return transform


    @staticmethod
    def transform_points(points, transform, img_size = None):
        points = np.float32(points)
        points = np.concatenate(
            (points, np.ones(shape=(len(points), 1), dtype=np.float32)),
            axis=1
        )

        if not img_size is None:
            img_w, img_h = img_size[0:2]
            points[:, 0] *= img_w
            points[:, 1] *= img_h

        points = transform.dot(points.T).T

        divide = points[:,2].reshape((points.shape[0]))
        points[:, 0:2] /= divide[:, np.newaxis]
        points[:, 0] /= img_w
        points[:, 1] /= img_h

        return points[:, 0:2].tolist()


    @staticmethod
    def transform_rectangle(rect, transform, img_size = None):
        x1, y1, x2, y2 = rect

        tl_p = [x1, y1]
        tr_p = [x2, y1]
        bl_p = [x1, y2]
        br_p = [x2, y2]

        points = np.float32([tl_p, tr_p, bl_p, br_p])
        points = np.concatenate(
            (points, np.ones(shape=(len(points), 1), dtype=np.float32)),
            axis = 1
        )

        if not img_size is None:
            im_w, im_h = img_size
            points[:,0] *= im_w
            points[:,1] *= im_h

        points = transform.dot(points.T).T

        divide = points[:,2].reshape((points.shape[0],))
        points[:,0:2] /= divide[:, np.newaxis]
        
        left    = points[:,0].min()
        right   = points[:,0].max()
        top     = points[:,1].min()
        bottom  = points[:,1].max()
        
        if not img_size is None:
            left    /= im_w
            top     /= im_h
            right   /= im_w
            bottom  /= im_h

        return [left, top, right, bottom]


def augmentation_rotate3d(data, angles, inter_type=cv.INTER_AREA):
    new_data = {}
    img = data['image']

    h, w = img.shape[0:2]

    transform = Augmentation3d.get_perspective_matrix(angles, (w, h))

    new_data['image'] = cv.warpPerspective(img, transform, (w, h), flags=inter_type)

    if 'points' in data:
        points = data['points']
        new_data['points'] = Augmentation3d.transform_points(points, transform, (w, h))

    if 'rectangles' in data:
        rectangles = data['rectangles']
        new_data['rectangles'] = delegate_aug_rectangles_from_points(
            rectangles,
            Augmentation3d.transform_points,
            transform, 
            (w, h)
        )

    return new_data


def augmentation_rotate3d_rect(rectangle, angles, normalize_size=(1.0, 1.0)):
    transform = Augmentation3d.get_perspective_matrix(angles)
    return Augmentation3d.transform_rectangle(
        rectangle, transform, normalize_size
    )
'''
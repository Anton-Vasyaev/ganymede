# project
import ganymede.math.point2 as point2
# 3rd party
import numpy as np


def delegate_aug_rectangles_from_points(
    rectangles,
    points_aug_function,
    *function_parameters
):
    aug_rectangles = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect

        points = [
            (x1, y1), # left top
            (x2, y1), # right top
            (x1, y2), # left bottom
            (x2, y2)  # right bottom
        ]

        aug_points = points_aug_function(points, *function_parameters)
        x1, y1, x2, y2 = point2.get_bbox(aug_points)

        aug_rectangles.append([x1, y1, x2, y2])

    return aug_rectangles



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
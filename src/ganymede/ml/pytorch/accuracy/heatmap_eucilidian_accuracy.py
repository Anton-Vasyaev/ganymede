from .euclidian_accuracy import euclidian_accuracy
from ganymede.ml.pytorch.post_processing import heatmap_keypoint_detection


def heatmap_euclidian_accuracy(output, target):
    output_kp = heatmap_keypoint_detection(output)
    target_kp = heatmap_keypoint_detection(target)

    return euclidian_accuracy(output_kp, target_kp)

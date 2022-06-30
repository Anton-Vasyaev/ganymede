def accuracy_iou(output, target):
    output = output > 0.9
    target = target > 0.9

    true_positives = (output == target) & (output == True)

    false_positives = (output == True) & (target == False)

    false_negatives = (output == False) & (target == True)

    tp = true_positives.sum()
    fp = false_positives.sum()
    fn = false_negatives.sum()
    
    return tp / (tp + fp + fn)
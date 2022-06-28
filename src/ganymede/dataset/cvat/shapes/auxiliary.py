def load_label_and_points_from_xml(
    root,
    normalize_size = (1.0, 1.0)
):
    norm_w, norm_h = normalize_size

    label = root.get('label')
    points_str = root.get('points')

    points_str_l = points_str.split(';')
    points = []
    for points_str in points_str_l:
        x, y = points_str.split(',')
        x, y = float(x) / norm_w, float(y) / norm_h

        points.append([x, y])

    return label, points
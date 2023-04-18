import ganymede.math.bbox2 as m_bbox2



b1 = (0.5, 0.1, 0.9, 0.8)
b2 = (0.4, 0.2, 0.8, 0.9)

print(m_bbox2.iou(b1, b2))
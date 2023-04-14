import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def get_object_bbox(seg_json, class_id):
    geo = seg_json["geo_transform"]
    boundary = seg_json["bbox"]
    polygons = seg_json["polygons"]
    labels = seg_json["labels"]

    bboxes = []
    for i in range(len(labels)):
        if labels[i] == class_id:
            polygon = np.array(polygons[i])
            [xmin, ymin] = np.amin(polygon, 0)
            [xmax, ymax] = np.amax(polygon, 0)

            if xmax < boundary[0] or ymin > boundary[1] or xmin > boundary[2] or ymax < boundary[3]:
                continue

            xmin = max(xmin, boundary[0])
            ymin = max(ymin, boundary[3])
            xmax = min(xmax, boundary[2])
            ymax = min(ymax, boundary[1])

            xmin_ = abs(int((xmin - boundary[0]) / geo[1]))
            ymax_ = abs(int((ymin - boundary[1]) / geo[5]))
            xmax_ = abs(int((xmax - boundary[0]) / geo[1]))
            ymin_ = abs(int((ymax - boundary[1]) / geo[5]))
            bboxes.append([xmin_, ymin_, xmax_, ymax_])
    return bboxes


def get_object_points(seg_mask, class_id):
    if len(seg_mask.shape) == 3:
        seg_mask = seg_mask[:, :, 0]
    result = {}
    segment_mask = np.where(seg_mask == class_id, class_id, 0)
    plt.imshow(segment_mask)
    plt.title('mask')
    plt.axis('off')
    plt.show()
    segment_mask = segment_mask.astype(np.uint8)  # 重要！！！
    '''
        num_labels：所有连通域的数目
        labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
        stats：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和连通域的面积
        centroids：连通域的中心点
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segment_mask, connectivity=8)
    result['num_labels'] = num_labels
    classes = []
    points = []
    for i in range(num_labels):
        if stats[i][4] > 100:
            [x, y] = centroids[i]
            x_ = int(x)
            y_ = int(y)
            if segment_mask[x_][y_] > 0:
                points.append([x_, y_])
                classes.append(segment_mask[x_][y_])

    result['labels'] = classes
    result['points'] = points

    return result


def main():
    root = 'data/image.json'
    with open(root, "r") as f:
        jn = json.load(f)
    bboxes = get_object_bbox(jn, 1)


if __name__ == '__main__':
    main()

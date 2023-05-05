import json
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

Cityscapes_classes = {'unlabeled': 0, 'ego vehicle': 1, 'rectification border': 2, 'out of roi': 3, 'static': 4,
                      'dynamic': 5, 'ground': 6, 'road': 7, 'sidewalk': 8, 'parking': 9, 'rail track': 10,
                      'building': 11, 'wall': 12, 'fence': 13, 'guard rail': 14, 'bridge': 15, 'tunnel': 16, 'pole': 17,
                      'polegroup': 18, 'traffic light': 19, 'traffic sign': 20, 'vegetation': 21, 'terrain': 22,
                      'sky': 23, 'person': 24, 'rider': 25, 'car': 26, 'truck': 27, 'bus': 28, 'caravan': 29,
                      'trailer': 30, 'train': 31, 'motorcycle': 32, 'bicycle': 33}

Cityscapes_available_classes = ['person', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=1))


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str):
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def get_geo_bbox(seg_json, padding=0):
    geo = seg_json["geo_transform"]
    boundary = seg_json["bbox"]
    polygons = seg_json["polygons"]
    labels = seg_json["labels"]

    width = abs(int((boundary[2] - boundary[0]) / geo[1]))
    height = abs(int((boundary[3] - boundary[1]) / geo[5]))

    bboxes = {}
    farmland = []
    greenhouse = []
    tree = []
    pond = []
    house = []
    for label, polygon in zip(labels, polygons):
        polygon = np.array(polygon)
        [xmin, ymin] = np.amin(polygon, 0)
        [xmax, ymax] = np.amax(polygon, 0)

        # if xmax < boundary[0] or ymin > boundary[1] or xmin > boundary[2] or ymax < boundary[3]:
        #     continue  # 判断整个框是不是在界外

        # 保证框全部在界内
        # xmin = max(xmin, boundary[0])
        # ymin = max(ymin, boundary[3])
        # xmax = min(xmax, boundary[2])
        # ymax = min(ymax, boundary[1])

        # 转像素坐标
        xmin_ = abs(int((xmin - boundary[0]) / geo[1]))
        ymax_ = abs(int((ymin - boundary[3]) / geo[5]))
        xmax_ = abs(int((xmax - boundary[0]) / geo[1]))
        ymin_ = abs(int((ymax - boundary[3]) / geo[5]))
        if (ymax_ - ymin_) * (xmax_ - xmin_) < 1000:  # 面积太小的框丢掉
            continue
        bbox = [max(xmin_ - padding, 0),
                max(ymin_ - padding, 0),
                min(xmax_ + padding, width),
                min(ymax_ + padding, height)]
        if label == 1:
            farmland.append(bbox)
        elif label == 2:
            greenhouse.append(bbox)
        elif label == 3:
            tree.append(bbox)
        elif label == 4:
            pond.append(bbox)
        elif label == 5:
            house.append(bbox)
    bboxes['farmland'] = farmland
    bboxes['greenhouse'] = greenhouse
    bboxes['tree'] = tree
    bboxes['pond'] = pond
    bboxes['house'] = house

    return bboxes


def get_Cityscapes_bbox(obj, width, height, padding=0):
    polygon = obj['polygon']
    polygon = np.array(polygon)
    [xmin, ymin] = np.amin(polygon, 0)
    [xmax, ymax] = np.amax(polygon, 0)
    bbox = [max(xmin - padding, 0),
            max(ymin - padding, 0),
            min(xmax + padding, width),
            min(ymax + padding, height-50)]
    return bbox


def get_bbox(seg_json, class_id):
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
    bboxes = get_geo_bbox(jn, 1)


if __name__ == '__main__':
    main()

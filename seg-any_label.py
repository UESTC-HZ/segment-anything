import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from load_model import load_predictor_model
from segment_anything import predictor

VIT_H = 'vit_h'
VIT_L = 'vit_l'
VIT_B = 'vit_b'

CLASS_NAMES = ['farmland', 'greenhouse', 'tree', 'pond', 'house']


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=1))


def get_object_bbox(seg_json, padding=None):
    if padding == None:
        padding = 0
    geo = seg_json["geo_transform"]
    boundary = seg_json["bbox"]
    polygons = seg_json["polygons"]
    labels = seg_json["labels"]

    width = abs(int((boundary[2] - boundary[0]) / geo[1]))
    height = abs(int((boundary[1] - boundary[3]) / geo[5]))

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

        if xmax < boundary[0] or ymin > boundary[1] or xmin > boundary[2] or ymax < boundary[3]:
            continue  # 判断整个框是不是在界外

        # 保证框全部在界内
        xmin = max(xmin, boundary[0])
        ymin = max(ymin, boundary[3])
        xmax = min(xmax, boundary[2])
        ymax = min(ymax, boundary[1])

        # 转像素坐标
        xmin_ = abs(int((xmin - boundary[0]) / geo[1]))
        ymax_ = abs(int((ymin - boundary[1]) / geo[5]))
        xmax_ = abs(int((xmax - boundary[0]) / geo[1]))
        ymin_ = abs(int((ymax - boundary[1]) / geo[5]))
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


def create_seg_laebl(root):
    image_path = os.path.join(root, "images")
    json_label_path = os.path.join(root, "jsons")
    segany_label_path = os.path.join(root, "seg-any_mask")
    if not os.path.exists(segany_label_path):
        os.mkdir(segany_label_path)
    sam = load_predictor_model(VIT_B)
    for img in os.listdir(image_path):
        # if img.endswith("2.jpg"):
        #     continue
        image = cv2.imread(os.path.join(image_path, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        print("height:" + str(h) + " width:" + str(w))
        json_name = img.replace('jpg', 'json')
        with open(os.path.join(json_label_path, json_name), "r") as f:
            jn = json.load(f)
        bboxes = get_object_bbox(jn, 5)

        class_mask = []
        for i in range(5):
            class_name = CLASS_NAMES[i]
            full_mask = np.zeros((h, w), dtype=bool)
            input_boxes = np.array(bboxes[class_name])
            if len(input_boxes) > 0:
                input_boxes = torch.tensor(input_boxes, device=sam.device)
                sam.set_image(image)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, image.shape[:2])

                masks, _, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                for mask in masks:
                    tmp_mask = mask.cpu().numpy()[0]
                    full_mask = full_mask | tmp_mask
                plt.imshow(full_mask)
                # for box in input_boxes:
                #     show_box(box.cpu().numpy(), plt.gca())
                # plt.show()
                full_mask, _ = remove_small_regions(full_mask, 1000, "holes")
                full_mask, _ = remove_small_regions(full_mask, 1000, "islands")
                # plt.imshow(full_mask)
                # plt.show()

            class_mask.append(full_mask)

        segany_mask = np.zeros((h, w))
        for i in range(len(class_mask)):
            mask = class_mask[i]
            mask = np.where(mask, i + 1, 0)
            segany_mask = np.add.reduce([segany_mask, mask])

            segany_mask = np.where(segany_mask > (i + 1), segany_mask - (i + 1), segany_mask)
        # print(np.unique(segany_mask))
        # plt.imshow(segany_mask)
        # plt.show()

        cv2.imwrite(os.path.join(segany_label_path, img.replace(".jpg", ".png")), segany_mask)


if __name__ == '__main__':
    root = 'D:\Desktop\classes_08\merge_house\compress_0.2_images_10'
    create_seg_laebl(root)

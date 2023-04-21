import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm

from label_process import get_geo_bbox, remove_small_regions, show_box
from load_model import load_predictor_model
from segment_anything import predictor

VIT_H = 'vit_h'
VIT_L = 'vit_l'
VIT_B = 'vit_b'
DATASET_TYPE = ['geo', 'Cityscapes', 'COCOstuff']

GEO_CLASS_NAMES = ['farmland', 'greenhouse', 'tree', 'pond', 'house']


def create_geo_segany_laebl(root, model_type=VIT_B):
    image_path = os.path.join(root, "images")
    json_label_path = os.path.join(root, "jsons")
    segany_label_path = os.path.join(root, "seg-any_mask")
    if not os.path.exists(segany_label_path):
        os.mkdir(segany_label_path)

    # 初始化segment-anything模型
    sam = load_predictor_model(model_type)

    for img in tqdm(os.listdir(image_path)):
        image = cv2.imread(os.path.join(image_path, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # print("height:" + str(height) + " width:" + str(width))

        json_name = img.replace('jpg', 'json')
        with open(os.path.join(json_label_path, json_name), "r") as f:
            jn = json.load(f)
        bboxes = get_geo_bbox(jn, 0)

        class_mask = []
        for i in range(len(GEO_CLASS_NAMES)):
            class_name = GEO_CLASS_NAMES[i]
            full_mask = np.zeros((h, w), dtype=bool)
            input_boxes = np.array(bboxes[class_name])
            if len(input_boxes) > 0:
                input_boxes = torch.tensor(input_boxes, device=sam.device)
                sam.set_image(image)
                transformed_boxes = sam.transform.apply_boxes_torch(input_boxes, (h, w))

                masks, _, _ = sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )

                for mask in masks:
                    tmp_mask = mask.cpu().numpy()[0]
                    full_mask = full_mask | tmp_mask
                # plt.figure(figsize=(10, 10))
                # plt.imshow(full_mask)
                # for box in input_boxes:
                #     show_box(box.cpu().numpy(), plt.gca())
                # plt.show()
                full_mask, _ = remove_small_regions(full_mask, 1000, "holes")
                full_mask, _ = remove_small_regions(full_mask, 1000, "islands")
                # plt.figure(figsize=(10, 10))
                # plt.imshow(full_mask)
                # plt.show()

            class_mask.append(full_mask)

        segany_mask = np.zeros((h, w))
        for i in range(len(class_mask)):
            mask = class_mask[i]
            mask = np.where(mask, i + 1, 0)
            segany_mask = np.add.reduce([segany_mask, mask])

            segany_mask = np.where(segany_mask > (i + 1), segany_mask - (i + 1), segany_mask)

        # 压缩后的图片大小
        h_ = h
        w_ = w
        if h > 1024 or w > 1024:
            if h >= w:
                h_ = 1024
                w_ = int(w * h_ / h)
            elif h < w:
                w_ = 1024
                h_ = int(h * w_ / w)
        segany_mask = cv2.resize(segany_mask, (w_, h_))
        # print(np.unique(segany_mask))
        # plt.figure(figsize=(10, 10))
        # plt.imshow(segany_mask)
        # plt.show()
        cv2.imwrite(os.path.join(segany_label_path, img.replace(".jpg", ".png")), segany_mask)


def create_Cityscapes_segany_laebl(root):
    image_path = os.path.join(root, "images")
    json_label_path = os.path.join(root, "labels")
    segany_label_path = os.path.join(root, "seg-any_labels")
    if not os.path.exists(segany_label_path):
        os.mkdir(segany_label_path)

    # 初始化segment-anything模型
    sam = load_predictor_model(VIT_B)

    for img in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        json_name = img.replace('png', 'json')
        with open(os.path.join(json_label_path, json_name), "r") as f:
            jn = json.load(f)
        w = jn['imgWidth']
        h = jn['imgHeight']


def create_COCOstuff_segany_laebl():
    pass


def create_segany_label(root, dataset_type, model_type):
    assert dataset_type in DATASET_TYPE, 'DataSet Type Error'
    if dataset_type is DATASET_TYPE[0]:
        create_geo_segany_laebl(root, model_type)
    elif dataset_type is DATASET_TYPE[1]:
        create_Cityscapes_segany_laebl(root, model_type)
    elif dataset_type is DATASET_TYPE[2]:
        create_COCOstuff_segany_laebl(root, model_type)


if __name__ == '__main__':
    root = 'data/compress_0.1_images_1'
    dataset_type = "geo"
    model_type = VIT_H
    create_segany_label(root, dataset_type, model_type)

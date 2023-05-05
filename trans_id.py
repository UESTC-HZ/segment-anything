import cv2
import os
from tqdm import tqdm

path = 'labels'
if os.path.exists(path):
    for img_name in tqdm(os.listdir(path)):
        if img_name.endswith('_gtCoarse_labelIds.png') or img_name.endswith('_gtFine_labelIds.png'):
            image = os.path.join(path, img_name)
            img = cv2.imread(image, 0)
            img[img == 7] = 0
            img[img == 8] = 1
            img[img == 11] = 2
            img[img == 12] = 3
            img[img == 13] = 4
            img[img == 17] = 5
            img[img == 19] = 6
            img[img == 20] = 7
            img[img == 21] = 8
            img[img == 22] = 9
            img[img == 23] = 10
            img[img == 24] = 11
            img[img == 25] = 12
            img[img == 26] = 13
            img[img == 27] = 14
            img[img == 28] = 15
            img[img == 31] = 16
            img[img == 32] = 17
            img[img == 33] = 18

            cv2.imwrite(os.path.join(path, img_name), img, )

path = 'seg-any_labels'
if os.path.exists(path):
    for img_name in tqdm(os.listdir(path)):
        if img_name.endswith('_gtCoarse_seg-any.png'):
            image = os.path.join(path, img_name)
            img = cv2.imread(image, 0)
            img[img == 7] = 0
            img[img == 8] = 1
            img[img == 11] = 2
            img[img == 12] = 3
            img[img == 13] = 4
            img[img == 17] = 5
            img[img == 19] = 6
            img[img == 20] = 7
            img[img == 21] = 8
            img[img == 22] = 9
            img[img == 23] = 10
            img[img == 24] = 11
            img[img == 25] = 12
            img[img == 26] = 13
            img[img == 27] = 14
            img[img == 28] = 15
            img[img == 31] = 16
            img[img == 32] = 17
            img[img == 33] = 18

            cv2.imwrite(os.path.join(path, img_name), img)

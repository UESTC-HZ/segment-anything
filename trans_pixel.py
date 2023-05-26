import os

import cv2
import numpy as np
from tqdm import tqdm

root = "data/COCOstuff/coco_mask"
for png in tqdm(os.listdir(root)):
    mask = cv2.imread(os.path.join(root, png), 0)
    mask = np.where(mask == 1, 255, 0)
    cv2.imwrite(os.path.join(root, png), mask)


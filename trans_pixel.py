import os

import cv2
import numpy as np
from tqdm import tqdm

root = "./labels"
for png in tqdm(os.listdir(root)):
    if png.find('seg'):
        os.rename(os.path.join(root, png), os.path.join(root, png.replace('seg', "seg_")))

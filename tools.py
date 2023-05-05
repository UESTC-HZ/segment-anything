import os
import shutil

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2


def check_label(root, check_image, check_label):
    images_path = os.path.join(root, check_image)
    check_path = os.path.join(root, check_label)

    for img in tqdm(os.listdir(images_path)):
        check = img.replace('jpg', 'png')
        image = cv2.imread(os.path.join(images_path, img))
        label = cv2.imread(os.path.join(check_path, check))
        if image.shape == label.shape:
            continue
        else:
            print(img + ':' + str(image.shape) + "!=" + check + ':' + str(label.shape))


def copy_to_trainval(root):
    seg_any_mask_path = os.path.join(root, 'seg-any_mask')
    segmentation_path = os.path.join(root, 'segmentation')

    seg_any_train = os.path.join(segmentation_path, 'train', 'seg-any_labels')
    if not os.path.exists(seg_any_train):
        os.mkdir(seg_any_train)
    seg_any_val = os.path.join(segmentation_path, 'val', 'seg-any_labels')
    if not os.path.exists(seg_any_val):
        os.mkdir(seg_any_val)

    train_list = os.listdir(os.path.join(segmentation_path, 'train', 'labels'))
    val_list = os.listdir(os.path.join(segmentation_path, 'val', 'labels'))

    for mask in tqdm(os.listdir(seg_any_mask_path)):
        if mask in train_list:
            shutil.copyfile(os.path.join(seg_any_mask_path, mask), os.path.join(seg_any_train, mask))
        elif mask in val_list:
            shutil.copyfile(os.path.join(seg_any_mask_path, mask), os.path.join(seg_any_val, mask))
        else:
            print(mask + ' not found in trainval!')


def show_image(root=None):
    if root == None:
        root = 'D:/Desktop/show'
    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(30, 42))
    fig.subplots_adjust(top=0.95)

    images = os.path.join(root, "images")
    # labels = os.path.join(root, "labels")
    inference = os.path.join(root, "inference")
    size = (700,800)

    for i, img in enumerate(os.listdir(images)):
        image_path = os.path.join(images, img)
        # label_path = os.path.join(labels, img.replace('.jpg', '.png'))
        inference_path = os.path.join(inference, img)

        img_a = Image.open(image_path).resize(size)
        # img_b = Image.open(label_path).resize(size)
        img_c = Image.open(inference_path).resize(size)

        axs[i][0].imshow(img_a)
        # axs[i][1].imshow(img_b)
        axs[i][1].imshow(img_c)
        axs[i][0].axis('off')
        axs[i][1].axis('off')
        # axs[i][2].axis('off')
        if i == 0:
            axs[i, 0].set_title('Image', fontsize=40)
            # axs[i, 1].set_title('Label', fontsize=40)
            axs[i, 1].set_title('Inference', fontsize=40)

    # root = 'D:/Desktop/bad_show'
    # images = os.path.join(root, "images")
    # # labels = os.path.join(root, "labels")
    # inference = os.path.join(root, "inference")
    #
    # for i, img in enumerate(os.listdir(images)):
    #     image_path = os.path.join(images, img)
    #     # label_path = os.path.join(labels, img.replace('.jpg', '.png'))
    #     inference_path = os.path.join(inference, img)
    #
    #     img_a = Image.open(image_path).resize(size)
    #     # img_b = Image.open(label_path).resize(size)
    #     img_c = Image.open(inference_path).resize(size)
    #
    #     axs[i][2].imshow(img_a)
    #     # axs[i][1].imshow(img_b)
    #     axs[i][3].imshow(img_c)
    #     axs[i][2].axis('off')
    #     axs[i][3].axis('off')
    #     # axs[i][2].axis('off')
    #     if i == 0:
    #         axs[i, 2].set_title('Image', fontsize=40)
    #         # axs[i, 1].set_title('Label', fontsize=40)
    #         axs[i, 3].set_title('Inference', fontsize=40)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    root = 'D:\Desktop\classes_08\merge_house\compress_0.1_images_1\merge_data'

    # check_label(root, 'images', 'labels')
    # copy_to_trainval(root)
    show_image()

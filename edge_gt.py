import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def get_edge(root, pic_name, save_root):
    # import ipdb;ipdb.set_trace()
    pic_name_1 = os.path.join(root, pic_name)
    img = cv2.imread(pic_name_1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dist = distance_transform_edt(img)
    dist[dist > 3] = 0
    dist = dist * 255
    img = cv2.cvtColor(dist.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    save_name = os.path.join(save_root, pic_name.split()[0])
    # import ipdb;ipdb.set_trace()
    cv2.imwrite(save_name, img)

def main():
    root = '/home/hao/文档/coco2017/Binary_map_aug'
    save_root = './edge_gt'
    os.mkdir(save_root)
    for pic_name in tqdm(os.listdir(root)):
        get_edge(root, pic_name,save_root)


if __name__ == '__main__':
    main()

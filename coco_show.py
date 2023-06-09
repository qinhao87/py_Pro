import cv2
import random
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

train_json = '/home/hao/桌面/RDSNet/data/coco/annotations/instances_val2017_w.json'
train_path = '/home/hao/文档/coco2017/val2017/'


def visualization_seg(num_image, json_path, img_path, str=' '):
    # 需要画图的是第num副图片, 对应的json路径和图片路径,
    # str = ' '为类别字符串，输入必须为字符串形式 'str'，若为空，则返回所有类别id
    coco = COCO(json_path)

    catIds = coco.getCatIds(catNms=['str'])  # 获取指定类别 id
    # import ipdb;ipdb.set_trace()
    imgIds = coco.getImgIds(catIds=catIds)  # 获取图片i
    # img = coco.loadImgs(imgIds[num_image - 1])[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    img = coco.loadImgs(139)[0]
    image = io.imread(train_path + img['file_name'])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    # 读取在线图片的方法
    # I = io.imread(img['coco_url'])
    # print(anns)

    plt.imshow(image)
    coco.showAnns(anns)
    plt.show()


if __name__ == "__main__":
    visualization_seg(139, train_json, train_path)


# from pycocotools.coco import COCO
# import os
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
# import cv2
#
# def convert_coco2mask_show(image_id):
#     img = coco.imgs[image_id]
#     # img = coco.loadImgs(192)[0]
#     # loading annotations into memory...
#     # Done (t=12.70s)
#     # creating index...
#     # index created!
#     # img
#     # {'license': 2,
#     #  'file_name': '000000000074.jpg',
#     #  'coco_url': # 'http://images.cocodataset.org/train2017/000000000074.jpg',
#     #  'height': 426,
#     #  'width': 640,
#     #  'date_captured': '2013-11-15 03:08:44',
#     #  'flickr_url': # 'http://farm5.staticflickr.com/4087/5078192399_aaefdb5074_z.jpg# ',
#     #  'id': 74}
#     image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
#     plt.imshow(image, interpolation='nearest')
#     plt.show()
#     plt.imshow(image)
#
#     cat_ids = coco.getCatIds()
#     anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
#     anns = coco.loadAnns(anns_ids)
#     coco.showAnns(anns)
#     mask = coco.annToMask(anns[0])
#     for i in range(len(anns)):
#         mask += coco.annToMask(anns[i])
#     plt.imshow(image, interpolation='nearest')
#     # plt.imshow(image)
#     # plt.imshow(mask)
#     # import ipdb;ipdb.set_trace()
#     mask = np.where(mask != 0, 1, 0)
#     mask_fore = cv2.cvtColor((mask[:,:,None]*image).astype(np.uint8),cv2.COLOR_BGR2GRAY)
#     # import ipdb;ipdb.set_trace()
#     grad_x = cv2.Sobel(mask_fore,-1,1,0)
#     grad_y = cv2.Sobel(mask_fore,-1,0,1)
#     grad = cv2.addWeighted(grad_x,0.5,grad_y,0.5,0)
#     # plt.imshow(grad)
#     cv2.imwrite("mask_fore.jpg", grad)
#     # plt.show()
#
#
# if __name__ == '__main__':
#     Dataset_dir = "/home/hao/文档/coco2017/"
#     coco = COCO(os.path.join(Dataset_dir, 'annotations/instances_train2017_w.json'))
#     img_dir = os.path.join(Dataset_dir, 'train2017')
#     save_dir = os.path.join(Dataset_dir, "Mask")
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)
#     image_id = 192
#     convert_coco2mask_show(image_id)

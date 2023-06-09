from pycocotools.coco import COCO
import cv2
import numpy as np

# 加载COCO数据集
dataDir = '/home/hao/文档/coco'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# 获取图像信息和标注信息
images_ids = coco.getImgIds()
# import ipdb;ipdb.set_trace()
# img_id = images_ids[0]  # 示例图像ID
for img_id in images_ids:
    img_info = coco.loadImgs(ids=[img_id])[0]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    ann_info = coco.loadAnns(ids=ann_ids)

    # 加载图像并将标注信息可视化
    img_path = '{}/{}/{}'.format(dataDir, dataType, img_info['file_name'])
    img = cv2.imread(img_path)
    mask_img = np.zeros(shape=img.shape[:2])
    # sobel = np.zeros(shape=img.shape[:2])
    for ann in ann_info:
        # 为每个标注区域创建掩膜
        mask = coco.annToMask(ann)
        # 将掩膜转换为二进制图像
        mask = np.array(mask, dtype=np.uint8)
        mask *= 255
        # import ipdb;ipdb.set_trace()
        # 显示mask

        mask_img += mask
        # mask_img[mask != 0] = (0, 255, 255)
        # alpha = 0.6
        # cv2.addWeighted(mask_img, alpha, img, 1 - alpha, 0, img)

    new_img = np.zeros_like(img)
    new_img[:, :, 0] = img[:, :, 0] * mask_img
    new_img[:, :, 1] = img[:, :, 1] * mask_img
    new_img[:, :, 2] = img[:, :, 2] * mask_img
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(new_img, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(new_img, -1, 0, 1, ksize=3)
    # sobel = cv2.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    sobel = np.where(sobel > 0, 255, 0)
    # edges = cv2.Canny(new_img.astype(np.uint8) * 255, 150, 200)
    edge_filename = './{}/{}'.format('edges', img_info['file_name'])
    # print(edge_filename)
    cv2.imwrite(edge_filename, sobel)
# # 显示图像和分割标注部分
# cv2.imshow('Image', mask_img)
# cv2.imshow('new_img', new_img)
# # cv2.imshow('Sobel', sobel)
# cv2.imshow('Canny', edges)
# cv2.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# img = cv2.imread("/home/hao/文档/HED-BSDS/train/aug_gt/0.0_1_0/2092.png")#img 三通道 np.uint8
# print(img.shape)
# cv2.namedWindow("img")
# cv2.imshow("img",img)
# cv2.waitKey()

img1 = cv2.imread("/home/hao/文档/coco/Binary_map_aug/1.png")#三通道 np.uint8
print(img1)
plt.imshow(img1)
plt.show()
# img_re = 1-img1
# print(img_re)
# _mask = np.argmax(img1,axis=0)
# _mask[_mask!=0]+=1
# mask_pad =np.pad(img1, mode='constant', constant_values=0)
# img_re = np.zeros(img1.shape)
# _,img_re = cv2.threshold(img1,0.1,255,cv2.THRESH_BINARY)
# import ipdb;ipdb.set_trace()
img_re =cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
dist = distance_transform_edt(img_re)

# print(dist)
dist[dist >3] = 0
print(dist.shape)
# plt.imshow(dist)
# plt.show()
dist= dist*255
# print(np.unique(dist))
img = cv2.cvtColor(dist.astype(np.uint8),cv2.COLOR_GRAY2BGR)
cv2.namedWindow('pic')
cv2.imshow("pic",img)
cv2.waitKey()
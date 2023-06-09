import cv2
import numpy as np

# 载入灰度原图，并且归一化
img_original = cv2.imread('/home/hao/桌面/RDSNet/data/coco/val2017/000000001000.jpg', 0) / 255
# 分别求X,Y方向的梯度
grad_X = cv2.Sobel(img_original, -1, 1, 0)
grad_Y = cv2.Sobel(img_original, -1, 0, 1)
# 求梯度图像
grad = cv2.addWeighted(grad_X, 0.5, grad_Y, 0.5, 0)
cv2.imshow('sobel', grad)
# cv2.imwrite("sobel.jpg",grad)
cv2.waitKey()
cv2.destroyAllWindows()
# img_original = cv2.imread('/home/hao/桌面/RDSNet/data/coco/val2017/000000001000.jpg', 0)
# # 求X方向梯度，并且输出图像一个为CV_8U,一个为CV_64F
# img_gradient_X_8U = cv2.Sobel(img_original, -1, 1, 0)
# img_gradient_X_64F = cv2.Sobel(img_original, cv2.CV_64F, 1, 0)
# # 将图像深度改为CV_8U
# img_gradient_X_64Fto8U = cv2.convertScaleAbs(img_gradient_X_64F)
# # 图像显示
# cv2.imshow('X_gradient_8U', img_gradient_X_8U)
# cv2.imshow('X_gradient_64Fto8U', img_gradient_X_64Fto8U)
# cv2.waitKey()
# cv2.destroyAllWindows()
# import cv2
# o=cv2.imread("/home/hao/桌面/RDSNet/data/coco/val2017/000000001000.jpg",cv2.IMREAD_GRAYSCALE)
# r1=cv2.Canny(o,100,200,L2gradient=True)
# r2=cv2.Canny(o,32,128)
# cv2.imshow("original",o)
# # cv2.imshow("result1",r1)
# cv2.imwrite("canny.jpg",r1)
# # print('覃皓 别学了 大卷王')
# # cv2.imshow("result2",r2)
# cv2.waitKey()
# cv2.destroyAllWindows()


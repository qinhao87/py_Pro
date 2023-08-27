import pandas as pd
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import radviz


# 初始化参数
def initalize_parameters(density_fanwei, img_shape, xmin, ymin, xmax, ymax):
    np.random.seed(2)

    density = np.random.randint(density_fanwei[0], density_fanwei[1] + 1, 1)
    pre_xmin = np.random.randint(1, img_shape[0], 1)
    pre_xmax = np.random.randint(pre_xmin + 1, img_shape[0] + 1, 1)
    pre_ymin = np.random.randint(1, img_shape[1], 1)
    pre_ymax = np.random.randint(pre_ymin + 1, img_shape[1] + 1, 1)

    # 字典存储参数
    parametrs = {'desity': density, 'pre_xmin': pre_xmin, 'pre_xmax': pre_xmax,
                                    'pre_ymin': pre_ymin, 'pre_ymax': pre_ymax}

#前向传播
def forward_propagation(input_imgs, parameters):
    desity = parameters['desity']
    pre_xmin = parameters['pre_xmin']
    pre_xmax = parameters['pre_xmax']
    pre_ymin = parameters['pre_ymin']
    pre_ymax = parameters['pre_ymax']

    #前向传播取ROI


# 建立神经网络
def nn_model(input_imgs, output_ROIs, Y, density_fanwei, xmin, ymin, xmax, ymax, n_h, num_iteratiosn=10000,
             print_cost=False):
    np.random.seed(3)

    # 初始化参数
    parameters = initalize_parameters(density_fanwei, input_imgs.shape, xmin, ymin, xmax, ymax)

    #梯度下降
    for i in range(0,num_iteratiosn):

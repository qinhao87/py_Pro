from pycocotools.coco import COCO
from operator import itemgetter
import json
import time
import shutil
import os
from collections import defaultdict
import json
from pathlib import Path
# dataDir = '.'
# dataType = 'val2017'
# # dataType='train2017'
# annFile = '{}/mini_instances_{}.json'.format(dataDir, dataType)
#
# # initialize COCO api for instance annotations
# coco = COCO(annFile)
#
# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# cat_nms = [cat['name'] for cat in cats]
# print('number of categories: ', len(cat_nms))
# print('COCO categories: \n', cat_nms)
#
# # 统计各类的图片数量和标注框数量
# # import ipdb;ipdb.set_trace()
# type_tuple = list()
# for cat_name in cat_nms:
#     catId = coco.getCatIds(catNms=cat_name)  # 1~90
#     imgId = coco.getImgIds(catIds=catId)  # 图片的id
#     annId = coco.getAnnIds(catIds=catId)  # 标注框的id
#
#     img_nums = tuple([cat_name, len(annId)])
#     type_tuple.append(img_nums)
#     # print('catId:', cat_name)
#     # print('imgId:', len(imgId))
#     # print('annId:', len(annId))
#
# types = sorted(type_tuple, key=itemgetter(1), reverse=True)
# print(types)

def stactics(dataDir, dataType):

    annFile = '{}/mini_instances_{}.json'.format(dataDir, dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat['name'] for cat in cats]
    type_tuple = list()
    # import ipdb;ipdb.set_trace()
    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=cat_name)  # 1~90
        if len(catId)>1:
            # print(cat_name)
            # print(catId)
            catId = catId[-1]
        imgId = coco.getImgIds(catIds=catId)  # 图片的id
        annId = coco.getAnnIds(catIds=catId)  # 标注框的id

        img_nums = tuple([cat_name, len(annId)])
        type_tuple.append(img_nums)
        # print('catId:', cat_name)
        # print('imgId:', len(imgId))
        # print('annId:', len(annId))
    # import ipdb;
    # ipdb.set_trace()
    types = sorted(type_tuple, key=itemgetter(1), reverse=True)
    # types = {i[0][0]:num for num, i in enumerate(sorted(type_tuple, key=itemgetter(1), reverse=True)[:20], 1)}
    print(len(types))
    return types

dataDir = '.'
dataType='val2017_1'
# import ipdb;ipdb.set_trace()
train = stactics(dataDir, dataType)
print(train)
# train = {1: 1, 3: 2, 62: 3, 84: 4, 44: 5, 47: 6, 67: 7, 51: 8, 10: 9, 28: 10, 31: 11, 16: 12, 9: 13, 15: 14, 8: 15, 20: 16, 4: 17, 64: 18, 38: 19, 27: 20}
# val ='.'
# dataType='val2017'
# val = stactics(val, dataType)
#
# for cla in train:
#     if cla not in val:
#         print(cla)
# class COCO_1:
#     def __init__(self, annotation_file=None, origin_img_dir="", types=None):
#         """
#         Constructor of Microsoft COCO helper class for reading and visualizing annotations.
#         :param annotation_file (str): location of annotation file
#         :param image_folder (str): location to the folder that hosts images.
#         :return:
#         """
#         # load dataset
#         self.origin_dir = origin_img_dir
#         self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()  # imgToAnns　一个图片对应多个注解(mask) 一个类别对应多个图片
#         self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
#         self.cla = types
#         if not annotation_file == None:
#             print('loading annotations into memory...')
#             tic = time.time()
#             dataset = json.load(open(annotation_file, 'r'))
#             assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
#             print('Done (t={:0.2f}s)'.format(time.time() - tic))
#             self.dataset = dataset
#             self.createIndex()
#
#     def createIndex(self):
#         # create index　　  给图片->注解,类别->图片建立索引
#         print('creating index...')
#         anns, cats, imgs = {}, {}, {}
#         imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
#         if 'annotations' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 imgToAnns[ann['image_id']].append(ann)
#                 anns[ann['id']] = ann
#
#         if 'images' in self.dataset:
#             for img in self.dataset['images']:
#                 imgs[img['id']] = img
#         # import ipdb;ipdb.set_trace()
#         if 'categories' in self.dataset:
#             for num,cat in enumerate(self.dataset['categories'], 1):
#                 if cat['id'] in self.cla.keys():
#                     # print(train)
#                     cat['id'] = num
#                     cats[cat['id']] = cat
#
#         if 'annotations' in self.dataset and 'categories' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 if ann['category_id'] in self.cla.keys():
#                     ann['category_id'] = self.cla[ann['category_id']]
#                     catToImgs[ann['category_id']].append(ann['image_id'])
#
#         print('index created!')
#
#         # create class members
#         self.anns = anns
#         self.imgToAnns = imgToAnns
#         self.catToImgs = catToImgs
#         self.imgs = imgs
#         self.cats = cats
#
#     def build(self, tarDir=None, tarFile='./new.json', N=1000):
#
#         load_json = {'images': [], 'annotations': [], 'categories': [], 'type': 'instances', "info": {"description": "This is stable 1.0 version of the 2014 MS COCO dataset.", "url": "http:\/\/mscoco.org", "version": "1.0", "year": 2014, "contributor": "Microsoft COCO group", "date_created": "2015-01-27 09:11:52.357475"}, "licenses": [{"url": "http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc\/2.0\/", "id": 2, "name": "Attribution-NonCommercial License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nc-nd\/2.0\/",
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               "id": 3, "name": "Attribution-NonCommercial-NoDerivs License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by\/2.0\/", "id": 4, "name": "Attribution License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-sa\/2.0\/", "id": 5, "name": "Attribution-ShareAlike License"}, {"url": "http:\/\/creativecommons.org\/licenses\/by-nd\/2.0\/", "id": 6, "name": "Attribution-NoDerivs License"}, {"url": "http:\/\/flickr.com\/commons\/usage\/", "id": 7, "name": "No known copyright restrictions"}, {"url": "http:\/\/www.usa.gov\/copyright.shtml", "id": 8, "name": "United States Government Work"}]}
#         if not Path(tarDir).exists():
#             Path(tarDir).mkdir()
#
#         for i in self.imgs:
#             if(N == 0):
#                 break
#             tic = time.time()
#             img = self.imgs[i]
#             load_json['images'].append(img)
#             fname = os.path.join(tarDir, img['file_name'])
#             anns = self.imgToAnns[img['id']]
#             for ann in anns:
#                 # import ipdb;ipdb.set_trace()
#                 if ann['category_id'] in self.cla.keys():
#                     ann['category_id'] = self.cla[ann['category_id']]
#                     load_json['annotations'].append(ann)
#             if not os.path.exists(fname):
#                 shutil.copy(self.origin_dir+'/'+img['file_name'], tarDir)
#             print('copy {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))
#             N -= 1
#         for i in self.cats:
#             # import ipdb;ipdb.set_trace()
#             load_json['categories'].append(self.cats[i])
#         with open(tarFile, 'w+') as f:
#             json.dump(load_json, f, indent=4)
#
#
# coco = COCO_1('/home/hao/py_Pro/mini_instances_train2017_1.json',
#             origin_img_dir='/home/hao/文档/coco2017/train2017', types=train)               # 完整的coco数据集的图片和标注的路径
# coco.build('./mini_train2017', './mini_instances_train2017.json', 29569)  # 保存图片路径
#
#
# coco = COCO_1('/home/hao/py_Pro/mini_instances_val2017_1.json',
#             origin_img_dir='/home/hao/文档/coco2017/val2017', types=train)                 # 完整的coco数据集的图片和标注的路径
# coco.build('./mini_val2017', './mini_instances_val2017.json', 1251)       # 保存图片路径
# print(train)
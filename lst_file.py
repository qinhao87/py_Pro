import os
from tqdm import tqdm


def train_lst(train_root):
    with open('./train_pair.lst','a+') as file:
        for pic_name in tqdm(os.listdir(train_root)):
            # import ipdb;ipdb.set_trace()
            context = os.path.join(train_root, pic_name) +' '+\
                      os.path.join('/home/hao/文档/hed-bsd/train/edge_gt', pic_name.split('.')[0]+'.png')
            file.write(context+'\n')

def val_lst(test_root):
    with open('./test.lst','a+') as file:
        for pic_name in tqdm(os.listdir(test_root)):
            # import ipdb;ipdb.set_trace()
            context = os.path.join(test_root, pic_name)
            file.write(context+'\n')

def main():
    # train_lst('/home/hao/文档/hed-bsd/train/JPEGImages')
    val_lst('/home/hao/文档/hed-bsd/test')
if __name__ == '__main__':
    main()
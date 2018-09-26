import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import random


class DefectsData(data.Dataset):
    """
    目录结构:  
    ./data/train/label+id.jpg
    ./data/test/id.jpg ; 
    
    number of lables = 11(包含'其他‘)+1(normal);
    """
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test 
        split_rate = 0.8
        
        imgs = []
        for root, dirs, files in os.walk(root):
            for file in files:
                imgs.append(os.path.join(root,file))
        tmp = [i for i in imgs if i.split('.')[-1] == 'jpg']
        imgs = tmp[:]
        imgs_num = len(imgs)
        
        # shuffle 
        random.seed(666)
        random.shuffle(imgs)
        
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(split_rate*imgs_num)]
        else:
            self.imgs = imgs[int(split_rate*imgs_num):]
            
        
        if transforms is None:
            
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(480),
                    T.CenterCrop(480),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(500), 
                    T.RandomResizedCrop(480),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
            
                    # T.RandomResizedCrop(960),
    def __getitem__(self, index):
        """
        class_dic = {'正常': 0, '涂层开裂': 1, '横条压凹': 2, '桔皮': 3, 
                   '擦花': 4, '漏底': 5, '凸粉': 6, '不导电': 7, '起坑': 8,
                   '碰伤':9, '脏点': 10, '其他': 11}
        """

        img_path = self.imgs[index]
        if self.test:
            # label = int(self.imgs[index].split('.')[-2].split('/')[-1])
            label = str(self.imgs[index].split('/')[-1])
        else:
            label = int(self.imgs[index].split('/')[-1].split('label')[0])
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
    	return len(self.imgs)

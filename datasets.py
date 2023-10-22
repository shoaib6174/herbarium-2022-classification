import os
import json
import random
import pandas as pd
from PIL import Image
from sklearn import preprocessing
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, DatasetFolder, default_loader

from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels         
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]


class GetDataTest(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform, Ids):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels   
        self.ids = Ids      
        
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
              
        return self.transform(x), self.fnames[index], self.ids[index]


def build_dataset(which_data, args):
    transform = build_transform(which_data, args)
    path = os.getcwd()
    parent = os.path.dirname(path)

    

    Base_DIR = f"{parent}/herbarium2022/"
    train_dir = Base_DIR + "train_images/"
    with open(Base_DIR + 'train_metadata.json', "r", encoding="ISO-8859-1") as file:
        train_meta = json.load(file)
    image_ids = [image["image_id"] for image in train_meta["images"]]
    image_dirs = [train_dir + image['file_name'] for image in train_meta["images"]]
    category_ids = [annotation['category_id'] for annotation in train_meta['annotations']]
    genus_ids = [annotation['genus_id'] for annotation in train_meta['annotations']]

    df = pd.DataFrame({
                        "image_id" : image_ids,
                        "image_dir" : image_dirs,
                        "category_id" : category_ids,
                        "genus" : genus_ids})

    nb_classes = len(df['category_id'].value_counts())
    le = preprocessing.LabelEncoder()
    le.fit(df['category_id'].values)

    
    print(nb_classes)
    X_Train, Y_Train = df['image_dir'].values, le.transform(df['category_id'].values)
    # le.inverse_transform([0, 0, 1, 2]) --> inverse transform
    
    train_full_data = GetData(train_dir, X_Train, Y_Train, transform)

    train_idx, val_idx = train_test_split(list(range(len(train_full_data))), test_size=.12, stratify=Y_Train)
    print(len(val_idx))
    print(len(train_idx))

    train_data = Subset(train_full_data, train_idx) 
    val_data = Subset(train_full_data, val_idx)

    # TEST DATA:
    
    test_dir = Base_DIR + "test_images/"
    with open(Base_DIR + 'test_metadata.json', "r", encoding="ISO-8859-1") as file:
        test_meta = json.load(file)
    test_ids = [image['image_id'] for image in test_meta]
    test_dirs = [test_dir + image['file_name'] for image in test_meta]
    df_test = pd.DataFrame({
                            "test_id" : test_ids,
                            "test_dir" : test_dirs
                        })
    # print(len(df_test))
    NUM_CL_test = len(df_test['test_dir'].value_counts())
    print(NUM_CL_test)
    X_Test = df_test['test_dir'].values
    Y_Test = None
    test_data =GetDataTest(test_dir, X_Test, Y_Test, transform, df_test['test_id'].values)

    if which_data=='train':
        dataset = train_data
    elif which_data=='val':
        dataset = val_data
    elif which_data=='test':
        dataset = test_data
    else:
        print('Wrong Dataset')
        exit()



    return dataset, nb_classes

def build_transform(which_data, args):
    resize_im = args.input_size > 32
    if which_data == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        # size = 448
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
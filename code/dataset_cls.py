import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

import numpy as np
class Dinov2DatasetTrain(Dataset):
    def __init__(self, train=True):
        if train:
            data_dir = ['./dinov2/org_feat_cls_new1', './dinov2/org_feat_cls_new2']
        else:
            data_dir = ['./dinov2/org_feat_cls_test']

        self.data_dir = data_dir
        self.file_list = []
        for dir in data_dir:
            
            self.file_list.extend(os.listdir(dir))
        


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        
        file_name = self.file_list[index]
        for data_dir in self.data_dir:
            if file_name in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file_name)
                feat = np.load(file_path)
                return feat, file_name

class Dinov2DatasetTest(datasets.ImageFolder):


    def __init__(
            self,
            root: str,
            transform=None,
            **kwargs
    ):
        super().__init__(
            root,
            transform,
            **kwargs
        )

    def __getitem__(self, index: int):
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


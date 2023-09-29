import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob

class TTLoader(Dataset):
    """
    is_train - 0 train, 1 val
    """
    def __init__(self, is_train=0, path='/workspace/P/tpy_data', transform=None):
        self.is_train = is_train
        #self.transform = transform
        if is_train ==0:
            self.path = glob.glob(path+'/train/*')
        else:
            self.path = glob.glob(path+'/test/*')

        self.totenosr = transforms.ToTensor()

    def __len__(self):
        return len(self.path)

    def _make_sqe(self, path):
        length = len(path)
        if length -20 > 0:
            start = np.random.randint(length-20)
        else:
            start = 0 

        obs = path[start:start+8]
        pred = path[start+8:start+8+12]

        return obs, pred

    def _preprocessor(self, path):
        imgs = []
        coords = []
        for i in path: # /workspace/P/normalized_data/geopotential_6hour/era5_z225_1973070712.npy
            geo_225 = np.load('/workspace/P/normalized_data/geopotential_6hour/era5_z225_'+i.split(' ')[5][:-1] + '.npy')
            geo_500 = np.load('/workspace/P/normalized_data/geopotential_6hour/era5_z500_'+i.split(' ')[5][:-1] + '.npy')
            geo_700 = np.load('/workspace/P/normalized_data/geopotential_6hour/era5_z700_'+i.split(' ')[5][:-1] + '.npy')
            uwind_225 = np.load('/workspace/P/normalized_data/uwind_6hour/era5_u225_'+i.split(' ')[5][:-1] + '.npy')
            uwind_500 = np.load('/workspace/P/normalized_data/uwind_6hour/era5_u500_'+i.split(' ')[5][:-1] + '.npy')
            uwind_700 = np.load('/workspace/P/normalized_data/uwind_6hour/era5_u700_'+i.split(' ')[5][:-1] + '.npy')
            vwind_225 = np.load('/workspace/P/normalized_data/vwind_6hour/era5_v225_'+i.split(' ')[5][:-1] + '.npy')
            vwind_500 = np.load('/workspace/P/normalized_data/vwind_6hour/era5_v500_'+i.split(' ')[5][:-1] + '.npy')
            vwind_700 = np.load('/workspace/P/normalized_data/vwind_6hour/era5_v700_'+i.split(' ')[5][:-1] + '.npy')
            img = np.stack([geo_225, geo_500, geo_700, uwind_225, uwind_500, uwind_700, vwind_225,vwind_500, vwind_700], axis=0) #
            imgs.append(img)
            coord = np.stack([float(i.split(' ')[2]), float(i.split(' ')[3])], axis=0)
            coords.append(coord)

        imgs = np.stack(imgs, axis=0)
        coords = np.stack(coords, axis=0)

        imgs = torch.from_numpy(imgs)
        coords = torch.from_numpy(coords)

        return imgs, coords


    def __getitem__(self, idx):
        clip_path = self.path[idx]

        with open(clip_path, 'r') as f:
            tc = f.readlines()

        #while(len(tc) < 20):
        #    tc.append(tc[-1])
        obs, future = self._make_sqe(tc)

        obs, coords1 = self._preprocessor(obs)
        future, coords2 = self._preprocessor(future)
        return obs, coords1.float(), future, coords2.float()


if __name__ == '__main__':
    loader = TTLoader()
    for idx, i in enumerate(loader):
        print(i[0].type(), i[1].type(), i[2].type(), i[3].type())

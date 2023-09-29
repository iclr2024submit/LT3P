import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob

class TLoader(Dataset):
    """
    is_train - 0 train, 1 val
    """
    def __init__(self, is_train=0, path='crop_typhoon/', transform=None):
        self.is_train = is_train
        self.transform = transform
        self.path = glob.glob(path+'*')*100
        self.totenosr = transforms.ToTensor()

    def __len__(self):
        return len(self.path)

    def _make_sqe(self, path):
        images = glob.glob(os.path.join(path,'*/*.npy'))
        images = [i for i in images if 'ir105' in i] ###
        images = sorted(images)
        s_index = np.random.randint(len(images)-20)

        return images[s_index:s_index+10], images[s_index+10:s_index+20]

    def _preprocessor(self, path, is_input=False):
        imgs = []
        if is_input:
            for i in path:
                ir = cv2.resize(np.load(i), (384,384))
                ir = (ir.astype(np.float32)-4562/1274)
                sw = cv2.resize(np.load(i.replace('ir105', 'sw038')), (384,384))
                sw = (sw.astype(np.float32)-15898/257)
                wv = cv2.resize(np.load(i.replace('ir105', 'wv063')), (384,384))
                wv = (wv.astype(np.float32)-3807/79)
                
                img = np.stack([ir, sw, wv], axis=0) # 
                imgs.append(img)
                # imgs.append((np.expand_dims(img, 0).astype(np.float32)-4562)/1274)
            imgs = np.stack(imgs, axis=0)
            imgs = torch.from_numpy(imgs)
        else:
            for i in path:
                img = np.load(i)
                img = cv2.resize(img, (384,384))
                imgs.append((np.expand_dims(img, 0).astype(np.float32)-4562)/1274)
            imgs = np.stack(imgs, axis=0)
            imgs = torch.from_numpy(imgs)
        return imgs

    def __getitem__(self, idx):
        clip_path = self.path[idx]
        inputs, outputs = self._make_sqe(clip_path)
        inputs = self._preprocessor(inputs, True)
        outputs = self._preprocessor(outputs)

        return inputs, outputs


if __name__ == '__main__':
    loader = TLoader()
    inputs, outputs = loader[0]
    print(inputs.shape, outputs.shape)

import torch
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import IAM4VP
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
from lt_tpc import TrajectoryTransformer
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob

def tensor_to_txt(output_tensor):
    # Convert tensor to CPU and detach from gradients
    output_tensor = output_tensor.cpu().detach().numpy()

    # Extract data and format it to a string list
    lines = []
    for idx, (x, y) in enumerate(output_tensor):
        line = f"{idx} {x} {y}\n"
        lines.append(line)

    return lines

def preprocessor(path):
    imgs = []
    coords = []

    for i  in path: # /workspace/P/normalized_data/geopotential_6hour/era5_z225_1973070712.npy
        i = i.replace('\t', ' ')
        print(i)
        geo_250 = np.load('./z250/z250_'+i.split(' ')[5][:-1] + '.npy')
        geo_500 = np.load('./z500/_z500_'+i.split(' ')[5][:-1] + '.npy')
        geo_700 = np.load('./z700/_z700_'+i.split(' ')[5][:-1] + '.npy')
        uwind_250 = np.load('./u250/_u250_'+i.split(' ')[5][:-1] + '.npy')
        uwind_500 = np.load('./u500/_u500_'+i.split(' ')[5][:-1] + '.npy')
        uwind_700 = np.load('./u700/_u700_'+i.split(' ')[5][:-1] + '.npy')
        vwind_250 = np.load('./v250/_v250_'+i.split(' ')[5][:-1] + '.npy')
        vwind_500 = np.load('./v500/_v500_'+i.split(' ')[5][:-1] + '.npy')
        vwind_700 = np.load('./v700/_v700_'+i.split(' ')[5][:-1] + '.npy')
        img = np.stack([geo_250, geo_500, geo_700, uwind_250, uwind_500, uwind_700, vwind_250,vwind_500, vwind_700], axis=0) #
        imgs.append(img)
        coord = np.stack([float(i.split(' ')[2]), float(i.split(' ')[3])], axis=0)
        coords.append(coord)

    imgs = np.stack(imgs, axis=0)
    coords = np.stack(coords, axis=0)

    imgs = torch.from_numpy(imgs)
    coords = torch.from_numpy(coords)

    return imgs, coords

import os
import matplotlib.pyplot as plt

def plot_and_save(output_lines, savepath):
    background_image = plt.imread('./background.png')

    x_data = []
    y_data = []
    ids = []
    for line in output_lines:
        columns = line.strip().split()
        coord_id = columns[0]
        x_value = float(columns[2]) * 15.566 + 135.051
        y_value = float(columns[1]) * 8.202 + 19.441
        x_data.append(x_value)
        y_data.append(y_value)
        ids.append(coord_id)

    plt.imshow(background_image, extent=[100, 180, 0, 60])
    plt.scatter(x_data[:8], y_data[:8], color='green', s=10)
    plt.scatter(x_data[8:20], y_data[8:20], color='red', s=10)
    plt.scatter(x_data[20:], y_data[20:], color='blue', s=10)

    plt.title(savepath.split('/')[-1].split('\\')[-1].split('_output.png')[0])
    plt.xlim(100, 180)
    plt.ylim(0, 60)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(savepath)
    plt.clf()

def read_and_plot_txt_file(filepath, savepath):
    background_image = plt.imread('./background.png')

    x_data = []
    y_data = []
    ids = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            columns = line.strip().split()
            coord_id = columns[0]
            x_value = float(columns[1]) * 15.566 + 135.051
            y_value = float(columns[0]) * 8.202 + 19.441
            x_data.append(x_value)
            y_data.append(y_value)
            ids.append(coord_id)

    plt.imshow(background_image, extent=[100, 180, 0, 60])

    plt.scatter(x_data[:8], y_data[:8], color='green', s=10)

    plt.scatter(x_data[8:], y_data[8:], color='red', s=10)


    plt.title(filepath.split('/')[-1].split('\\')[-1].split('.txt')[0])

    plt.xlim(100, 180)
    plt.ylim(0, 60)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.savefig(savepath)
    plt.clf()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_seq_len = 8
output_seq_len = 12
video_dim = 64
tensor_dim = 64  
embed_size = 64
num_layers = 1
heads = 4
model = TrajectoryTransformer(input_seq_len, output_seq_len, video_dim, tensor_dim, embed_size, num_layers, heads).to(device)
model = torch.nn.DataParallel(model)
cudnn.benchmark = True
checkpoint = torch.load('./1900.pth')
model.load_state_dict(checkpoint)

data = glob.glob('./2019/*')

for i in data:
    with open(i, 'r') as f:
        tc = f.readlines()
    if len(tc) == 20:
        img1, coords1 = preprocessor(tc[:8])
        img2, coords2 = preprocessor(tc[8:20])
        coords1 = coords1.float()
        coords2 = coords2.float()
        output = model(coords1.unsqueeze(0), img2.unsqueeze(0))
        output = output.squeeze(0)
        output = torch.cat([coords1.cpu(), output.cpu(), coords2.cpu()], dim=0)
        output = tensor_to_txt(output)
        with open(os.path.basename(i)[:-4]+'_pred.txt', 'w') as f:
            f.writelines(output)
        plot_and_save(output, os.path.basename(i)[:-3]+'png')

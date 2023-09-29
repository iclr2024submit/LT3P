import glob
import numpy as np

path = glob.glob('/workspace/IA4VPv2/data/crop_typhoon/*/*/*.npy')

mean_sum = 0
mean_std = 0
count = 0
for i in path:
    img = np.load(i)
    mean_sum+=np.mean(img)
    mean_std+=np.std(img) 
    count +=1
print(mean_sum/count, mean_std/count)


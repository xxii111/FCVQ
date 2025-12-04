
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from PIL import Image
import shutil

i=1
with open('nyu/nyu_test.txt') as f:
    image_list = f.readlines()
    for image in image_list:
        tem=image.split(' ')
        shutil.move('nyu/'+tem[0], f'NYUd_test/image/{i}.jpg')
        shutil.move('nyu/' + tem[1], f'NYUd_test/depth/{i}.png')
        i+=1




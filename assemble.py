import utillc
import sys, os, glob
import argparse
from collections import OrderedDict
from collections import defaultdict
import pickle
import cv2
import tqdm
import pandas
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils
import torch.multiprocessing as mp
import albumentations as A
import numpy as np
from glob import glob
import time
import datetime
import imageio
import logging
import glob
#from logging import info
import matplotlib.pyplot as plt 
import lzma
import os, gc
import psutil
from itertools import groupby
import argparse
from torch.utils.tensorboard import SummaryWriter
from itertools import islice
from logging import info
logging.basicConfig(level=logging.INFO,
                    format='%(pathname)s:%(lineno)d: [%(asctime)ss%(msecs)03d]:%(message)s',
                    datefmt='%Hh%Mm%S')

from utillc import *
import datetime

utillc.tempDir = "/mnt/hd1/tmp"
writer = SummaryWriter()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_folder", default=".")
    args = parser.parse_known_args()[0]
    root = args.data_folder
    pth = os.path.join(root, "train_v2", "train", "*.jpg")
    csv = os.path.join(root, "written_name_train_v2.csv")

    meta_day = pandas.read_csv(csv)
    EKOX(meta_day.head())

    image = cv2.imread(os.path.join(root, "train_v2", "train", "TRAIN_0001.jpg"))
    image = cv2.imread(pth)    
    plt.imshow(image); plt.show()
    
    EKOX(pth)
    images_path = glob.glob(pth)
    EKOX(len(images_path))
    def read(pth) :
        image = cv2.imread(pth)
        return image
    l = map(read, images_path)
    whf = lambda e : (e.shape[0], e.shape[1]) 
    
    x = list(islice(map(whf, l), 1122))
    #EKOX(x)
    #plt.hist([e[0] for e in x]);     plt.show()
    #plt.hist([e[1] for e in x]);     plt.show()
    EKO()

    
    
    
    l = list(map(read, images_path[0:3000]))
    EKO()
    np.random.shuffle(l)
    EKO()
    W, H = 1000, 1000
    im = np.ones((H,W,3)).astype(int) * 255
    n, tl, mh = 0, (0, 0), 0
    for i,e in enumerate(l) :
        h,w = whf(e)
        tly, tlx = tl
        if tlx + w > W :
            tlx, tly = 0, tly + mh
            mh = 0
            if tly + h > H :
                #plt.imshow(im); plt.show()
                ptho = os.path.join(root, "synth", "n_%04d.jpg" % n)
                cv2.imwrite(ptho, im)
                im = np.ones((H,W,3)).astype(int) * 255
                tl, mh = (0, 0), 0
                n += 1

                
        mh = max(mh, h)
        #EKON(e.shape, h, w, tlx, tly)
        #plt.imshow(e); plt.show()
        assert(tlx+w < W)
        if tly+h < H :
            im[tly:tly+h, tlx:tlx+w, :] = e
            tl = (tly, tlx+w)
            tly, tlx = tl        

        
        
        
    
        
    
    
    
    


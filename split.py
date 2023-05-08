
import ultralytics
ultralytics.checks()
from functools import partial
import re, json
import os
import argparse
import torch
from utillc import *
import numpy as np

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
from ultralytics import YOLO

import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data_folder", default="./synth")
parser.add_argument("--train_folder", default="/mnt/hd1/data/manuscript/train_folder")
args = parser.parse_known_args()[0]

data_folder = os.path.abspath(args.data_folder)
files = glob.glob(os.path.join(data_folder, "n*.png"))
np.random.shuffle(files)
n = len(files)
EKOX(n)
train, test, valid = files[: n*2//3], files[n*2//3 :  n*2//3 + n*2//6],  files[  n*2//3 + n*2//6 :]

os.makedirs(args.train_folder, exist_ok=True)
def mkd(ff_sset) :
    ff, sset = ff_sset
    fldr = os.path.join(args.train_folder, ff)
    os.makedirs(fldr, exist_ok=True)
    def link(f) :
        os.symlink(f, os.path.join(fldr, os.path.basename(f)))
    list(map(link, sset))
    list(map(link, [ e.replace("png", "txt") for e in sset]))
    return fldr
train_f, test_f, valid_f = list(map(mkd, zip([ 'train', 'test', 'valid'],
                                             (train, test, valid))))

with open('manuscript.yaml', "w") as fd :
    c = """
path: %s
train: %s
test:  %s
valid: %s

#Classes
nc: 1

#classes names
names: ['baptise']
""" % (args.train_folder, train_f, test_f, valid_f)
    fd.write(c)





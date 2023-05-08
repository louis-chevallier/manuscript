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
from typing import Type
from utillc import *

@torch.jit.script
def g(m : torch.Tensor) -> torch.Tensor :
    return m @ m

@torch.jit.script
def f() :
    dev = 'cpu'
    #dev = 'cuda'
    m = torch.ones((100, 100)).to(dev)
    for i in range(10000) :
        m = g(m)
EKO()
f()
EKO()
f()
EKO()
f()
EKO()
f()
EKO()


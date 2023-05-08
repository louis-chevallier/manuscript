
import ultralytics
ultralytics.checks()

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

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from scratch
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='manuscript.yaml',
                      imgsz=512,
                      device=0,
                      project='manuscript',
                      epochs=3)  # train the model




results = model.val()  # evaluate model performance on the validation set
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
success = model.export(format='onnx')  # export the model to ONNX format
     

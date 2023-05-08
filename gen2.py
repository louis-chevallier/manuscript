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

import handwriting_synthesis.data
from handwriting_synthesis import data, utils, models, callbacks
from handwriting_synthesis.sampling import HandwritingSynthesizer
import words

EKOX(len(words.words))
words_list = words.words
model_path="../pytorch-handwriting-synthesis-toolkit/checkpoints/Epoch_56"
device = torch.device("cuda")
bias = 0.2
thickness = 12
EKO()
synthesizer = HandwritingSynthesizer.load(model_path, device, bias)
synth2 = HandwritingSynthesizer.load(model_path, device, 0.3)
output_dir = "."
EKO()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data_folder", default="")
parser.add_argument("--frm", default=0, type=int)
args = parser.parse_known_args()[0]
root = args.data_folder


def points_to_image(points, shrink=0.2, thickness=3) :
    w, h = points['width'] * shrink, points['height'] * shrink
    image = np.ones((int(h), int(w), 3)).astype(np.uint8) * 255
    p = points["points"]
    for s in p :
        #EKOX(TYPE(s))
        p0 = tuple(map(int, np.asarray(s[0]) * shrink))
        for xy in s[1:] :
            xy = tuple(map(int, np.asarray(xy)*shrink))
            cv2.line(image, p0, xy, (0,0,0), thickness)
            p0 = xy
    return image


def gen_word(n, w=None, thickness=1, synth=synthesizer) :
    if w is None :
        i = np.random.randint(0, len(words_list))
        w = words_list[i]
    #EKO()
    seq = synth.generate_handwriting2(w)
    #EKO()
    points = utils.get_points(seq,
                              horizontal_padding=100,
                              vertical_padding=5)
    im =  points_to_image(points, shrink=0.06, thickness=thickness)
    #EKO()
    return im


def gen_pages() :
    W, H = 512, 512
    im = np.ones((H,W,3)).astype(np.uint8) * 255
    imR = np.ones((H,W,3)).astype(np.uint8) * 255
    nnn, n, tl, mh = 0, args.frm, (0, 0), 0
    wrds = map(gen_word, range(100000))
    whf = lambda e : (e.shape[0], e.shape[1]) 
    ann=[]
    for i,e1 in enumerate(wrds) :
        #EKO()
        bpts = np.random.randint(0, 20) == 1
        if bpts :
            e = gen_word(0, 'Baptise', synth=synth2)
        else :
            e =  e1
        #EKO()
        h,w = whf(e)
        # on veut des dessins de mots raisonnables, il arrive qu'ils soient bizarres ..
        if h < H / 4 and w < W / 4 :

            h,w = whf(e)
            tly, tlx = tl
            if tlx + w >= W :
                tlx, tly = 0, tly + mh
                mh = 0
                if tly + h > H :
                    #plt.imshow(im); plt.show()
                    ptho = os.path.join(root, "synth", "n_%04d.png" % n)
                    pthoR = ptho.replace('n_', 'Rn_')
                    ptho_json = ptho.replace('png', 'json')
                    ptho_txt = ptho.replace('png', 'txt')
                    cv2.imwrite(ptho, im)
                    cv2.imwrite(pthoR, imR)                
                    json_object = json.dumps({ "boxes" : ann}, indent=4)
                    with open(ptho_json, "w") as outfile:
                        outfile.write(json_object)
                    #EKOX(ann)
                    def line(a) :
                        #EKOX(a)
                        x,y,w,h = a[0][0], a[0][1], (a[1][0] - a[0][0]), (a[1][1] - a[0][1])
                        return '\t'.join(map(str, [0, x, y, w, h])) 
                    lines = map(line, ann)
                    with open(ptho_txt, "w") as outfile:
                        outfile.write('\n'.join(lines))

                    EKON(nnn)
                    EKOI(im)
                    ann = []
                    im = np.ones((H,W,3)).astype(np.uint8) * 255
                    imR = np.ones((H,W,3)).astype(np.uint8) * 255
                    nnn, tl, mh = 0, (0, 0), 0
                    tly, tlx = tl
                    n += 1
            mh = max(mh, h)
            #EKON(e.shape, h, w, tlx, tly)
            #plt.imshow(e); plt.show()

            if tlx+w >= W :
                EKON(nnn, tlx, tly, n, h, w, tl, W, H)

            #assert(tlx+w < W) # il arrive que le mot généré soit plus large que la page ..

            if tly+h < H :
                im[tly:tly+h, tlx:tlx+w, :] = e
                imR[tly:tly+h, tlx:tlx+w, :] = e
                if bpts :
                    #EKON(tlx+1, tly+1, tlx+w-1, tly+h)
                    cv2.rectangle(imR, (tlx+1, tly+1), (tlx+w-1, tly+h-1), (220,0,0), 1)
                    ann += [[  (tlx+1, tly+1), (tlx+w-1, tly+h-1) ]]

                tl = (tly, tlx+w)
                tly, tlx = tl
                nnn += 1
        #EKO()

    
gen_pages()

if False :

    for nn in range(12) :
        seq = synthesizer.generate_handwriting2("Baptises")
        points = utils.get_points(seq)
        EKOX(len(points['points']))
        EKOX(points['width'])
        EKOX(len(points['points'][0]))

        im = points_to_image(points)
        EKOI(im)

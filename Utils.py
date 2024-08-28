import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random
import pandas as pd


def shift_back(x, len_shift=2, bands=28):
    _, _, row, _ = x.shape
    for i in range(bands):
        x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
    return x[:, :, :, :row]

def shift_4(f, len_shift=0):
    [bs, nC, row, col] = f.shape
    shift_f = torch.zeros(bs, nC, row, col + (nC - 1) * len_shift).cuda().float()
    for c in range(nC):
        shift_f[:, c, :, c * len_shift:c * len_shift + col] = f[:, c, :, :]
    return shift_f


def shift_3(f, len_shift=0):
    [nC, row, col] = f.shape
    shift_f = torch.zeros(nC, row, col + (nC - 1) * len_shift).cuda()
    for c in range(nC):
        shift_f[c, :, c * len_shift:c * len_shift + col] = f[c, :, :]
    return shift_f


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist


def prepare_data(file_list, file_num):
    #HSI = np.zeros((((512, 512, 28, file_num))))
    HSI = np.zeros((((1024, 1024, 28, file_num))))
    for idx in range(file_num):
        path1 = file_list[idx]
        data = sio.loadmat(path1)
        HSI[:, :, :, idx] = data['img_expand'] / 65535.0
        #HSI[:, :, :, idx] = data['data_slice'] / 65535.0
    HSI[HSI < 0.] = 0.
    HSI[HSI > 1.] = 1.
    return HSI


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pkl'))
    # file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            # result = re.findall(".*model_(.*).pth.*", file_)
            result = re.findall(".*model_(.*).pkl.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    if err < 1.0e-10:
        return 100
    else:
        return 10 * np.log10((data_range ** 2) / err)

def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2



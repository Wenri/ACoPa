import torch
from itertools import pairwise

import matlabengine
import numpy as np
from einops import rearrange

from color_utils import read_image
from matlabengine import MatEng

eng = MatEng()


def get_pixels(p, start, end):
    mask = np.logical_and(p[:, 1] >= start, p[:, 1] < end)
    return p[mask]


def get_meancolor(p, start, end):
    mask = np.logical_and(p[:, 2] >= start, p[:, 2] < end)
    return np.mean(p[mask], axis=0)


def seg_hist(H, e=0.5):
    [idx], = eng.FTC_Seg(H[None].astype(np.double), e)
    return idx.astype(np.int_)


def operate_pixels(p):
    H, edges = np.histogram(p[:, 2], bins=256, range=(-128, 128))
    for [start, end] in pairwise(edges[seg_hist(H)]):
        c = get_meancolor(p, start, end)
        print(c)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    img = read_image(name).numpy()
    H, edges = np.histogram(img[:, 0], bins=256, range=(0, 2 * torch.pi))

    for [start, end] in pairwise(edges[seg_hist(H)]):
        p = get_pixels(img, start, end)
        operate_pixels(p)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('../nerf-pytorch/data/nerf_llff_data/fern/images/IMG_4026.JPG')
    input('Press any key to exit')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

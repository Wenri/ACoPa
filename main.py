from collections import namedtuple

import torch
from itertools import pairwise, starmap

import matlabengine
import numpy as np
from einops import rearrange

from color_utils import read_image
from matlabengine import MatEng


class SegHist:
    eng = MatEng()

    def _get_mask(self, start, end):
        return np.logical_and(self.p >= start, self.p < end)

    def __init__(self, p, range, e):
        self.p = p
        self.H, self.edges = np.histogram(p, bins=256, range=range)
        [idx], = self.eng.FTC_Seg(self.H[None].astype(np.float_), float(e))
        self.idx = idx.astype(np.int_)

    def __iter__(self):
        return starmap(self._get_mask, pairwise(self.edges[self.idx]))


class ACoPe:
    _COPE_RET_T = namedtuple(typename='_COPE_RET_T', field_names=['iH', 'iS', 'n', 'c', 's'])

    def __iter__(self):
        eH, eS = self.e
        for iH, mH in enumerate(SegHist(self.img[:, 0], range=(0., 2 * torch.pi), e=eH)):
            lab = self.lab[mH]
            for iS, mS in enumerate(SegHist(self.img[mH, 1], range=(0., 1.), e=eS)):
                p = lab[mS]
                c = np.mean(p, axis=0)
                s = np.std(p, axis=0)
                yield self._COPE_RET_T(iH, iS, p.shape[0], c, s)

    def __init__(self, img, lab, e=(1000., 1000.)):
        self.img = img
        self.lab = lab
        self.e = e


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    img, lab = read_image(name)
    for i, j, n, c, s in ACoPe(img, lab):
        print(i, j, n,
              np.array2string(c, precision=2, separator=',', suppress_small=True),
              np.array2string(s, precision=2, separator=',', suppress_small=True), )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('../nerf-pytorch/data/nerf_llff_data/fern/images/IMG_4026.JPG')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from collections import namedtuple

import torch
from itertools import pairwise, starmap, chain

import matlabengine
import numpy as np
from einops import rearrange

from color_utils import read_image
from matlabengine import MatEng

eng = MatEng()


class SegHist:
    def _get_mask(self, start, end):
        if start < end:
            return np.logical_and(self.p >= start, self.p < end)
        else:
            return np.logical_or(self.p >= start, self.p < end)

    def __init__(self, p, range, e, debug=False):
        self.p = p
        self.H, self.edges = np.histogram(p, bins=256, range=range)
        idx = eng.FTC_Seg(self.H[None].astype(np.float_), float(e), debug).result()
        self.idx = np.asarray(idx, dtype=np.int_).reshape(-1)

    def __iter__(self):
        edges = self.edges[self.idx]
        edges = chain(edges, edges[:1])
        return starmap(self._get_mask, pairwise(edges))


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

    def get_modes(self):
        w1 = list()
        f1 = list()
        for i, j, n, c, s in self:
            w1.append(n)
            f1.append(np.concatenate((c, s)))
            print(i, j, n,
                  np.array2string(c, precision=2, separator=',', suppress_small=True),
                  np.array2string(s, precision=2, separator=',', suppress_small=True), )
        w1 = np.stack(w1, axis=0) / sum(w1)
        f1 = np.stack(f1, axis=0)
        return w1, f1


def solve_transfer(name, ref):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello, Underworld. {name} -> {ref}')  # Press Ctrl+F8 to toggle the breakpoint.

    img = ACoPe(*read_image(name))
    ref = ACoPe(*read_image(ref))

    w1, f1 = img.get_modes()
    w2, f2 = ref.get_modes()

    f, fval = eng.testemd(f1, f2, w1[:, None], w2[:, None], nargout=2)
    print(f'Minimal flow cost {fval}')
    ft = np.matmul(f.T, f2)
    print('New modes', np.array2string(ft[..., :3], precision=4, separator=',', suppress_small=True), )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solve_transfer('../nerf-pytorch/data/nerf_llff_data/fern/images/IMG_4026.JPG',
                   '../nerf-pytorch/data/nerf_real_360/vasedeck/images/IMG_8475.JPG')
    # '../nerf-pytorch/data/nerf_synthetic/materials/train/r_15.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

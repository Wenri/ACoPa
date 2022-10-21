import math
from collections import namedtuple
from itertools import pairwise, starmap, chain

import imageio
import matlab.engine
import numpy as np
import torch
from einops import rearrange
from kornia.color import lab_to_rgb

from color_utils import read_image, save_image
from imagewrap import _make_warp
from matlabengine import MatEng

eng = MatEng()


class SegHist:
    def _get_mask(self, start, end):
        if start < end:
            p = np.logical_and(self.p >= start, self.p < end)
        else:
            p = np.logical_or(self.p >= start, self.p < end)
        return p

    def __init__(self, p, range, e, debug=False):
        self.p = p
        self.H, self.edges = np.histogram(p, bins=256, range=range)
        idx = eng.FTC_Seg(self.H[None].astype(np.float_), float(e), debug).result()
        self.idx = np.asarray(idx, dtype=np.int_).reshape(-1)

    def __iter__(self):
        edges = self.edges[self.idx]
        edges = chain(edges, edges[:1])
        return filter(np.any, starmap(self._get_mask, pairwise(edges)))


class ACoPe:
    _COPE_RET_T = namedtuple(typename='_COPE_RET_T', field_names=['iH', 'iS', 'n', 'c', 's'])
    _PARAM_E_T = namedtuple(typename='_PARAM_E_T', field_names=['eH', 'eS'])

    def __iter__(self):
        eH, eS = self.e
        for iH, mH in enumerate(SegHist(self.img[:, 0], range=(0., 2 * math.pi), e=eH)):
            lab = self.lab[mH]
            for iS, mS in enumerate(SegHist(self.img[mH, 1], range=(0., 1.), e=eS)):
                p = lab[mS]
                c = np.mean(p, axis=0)
                s = np.std(p, axis=0)
                yield self._COPE_RET_T(iH, iS, p.shape[0], c, s)

    def __init__(self, img, lab, size=None, e=_PARAM_E_T(1000., 100.)):
        self.img = img
        self.lab = lab
        self.size = size
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
    ref = ACoPe(*read_image(ref), e=(100., 10.))

    w1, f1 = img.get_modes()
    w2, f2 = ref.get_modes()

    f, fval = eng.testemd(f1, f2, w1[:, None], w2[:, None], nargout=2)
    print(f'Minimal flow cost {fval}')
    ft = np.matmul(f.T, f2)
    ft /= np.sum(f, axis=0)[:, None]

    print('Mode transfer')
    for old, new in zip(f1[:, :3], ft[:, :3]):
        print(np.array2string(old, precision=4, separator=',', suppress_small=True), '->',
              np.array2string(new, precision=4, separator=',', suppress_small=True))

    from_points = np.ascontiguousarray(f1[:, 1:3])
    to_points = np.ascontiguousarray(ft[:, 1:3])
    a, b = _make_warp(from_points, to_points, img.lab[:, 1], img.lab[:, 2])
    save_image('transimg.png', img.lab[:, 0], a, b, img.size)

    L = img.lab[:, 0] * 20 / 51
    transimg = np.stack((L, a, b), axis=1)
    transimg = rearrange(transimg, '(h w) c ->1 c h w', w=img.size[0])
    transimg = lab_to_rgb(torch.as_tensor(transimg))
    transimg = rearrange(transimg.squeeze(0), 'c h w -> h w c')
    transimg = np.clip((transimg.numpy() * 255).round(), 0, 255)

    imageio.imwrite('transimg1.png', transimg.astype(np.uint8))
    print('Transfered image:', transimg.shape)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    solve_transfer('../nerf-pytorch/data/nerf_llff_data/fern/images/IMG_4026.JPG',
                   # '../nerf-pytorch/data/nerf_real_360/vasedeck/images/IMG_8475.JPG')
                   # '../nerf-pytorch/data/nerf_synthetic/materials/train/r_15.png')
                   '../nerf-pytorch/data/gbc_scannet/scene0521_00/images/1498.jpg')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import matlabengine
import numpy as np
from einops import rearrange

from color_utils import read_image
from matlabengine import MatEng

eng = MatEng()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    img = read_image(name)
    H, edges = np.histogram(img[:, 1], bins=256, range=(-128, 128))
    H = rearrange(H, 'bin -> 1 bin').astype(np.double)

    idx, = eng.FTC_Seg(H, 0.)
    print(idx)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('/md/NeRF_Data/nerf_llff_data/fern/images/IMG_4026.JPG')
    input('Press any key to exit')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

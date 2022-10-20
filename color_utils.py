import logging
from io import BytesIO

import torch
from PIL import Image, ImageCms
from einops import rearrange
import numpy as np
from kornia.color import rgb_to_hsv


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * img ** (1 / 2.4) - 0.055, 12.92 * img)
    img[img > 1] = 1  # "clamp" tonemapper
    return img


def read_image(img_path):
    log = logging.getLogger(__name__)
    # Read image
    with Image.open(img_path) as image:
        orig_icc = image.info.get('icc_profile')
        # Extract original ICC profile
        with BytesIO(orig_icc) as icc:
            orig_icc = ImageCms.ImageCmsProfile(icc)
        desc = ImageCms.getProfileDescription(orig_icc)

        # Plot image with original ICC profile
        log.debug('Original ICC profile: {}'.format(desc))

        # Create sRGB ICC profile and convert image to sRGB
        srgb_icc = ImageCms.createProfile('sRGB')
        img = ImageCms.profileToProfile(image, orig_icc, srgb_icc)

    img = torch.from_numpy(np.asarray(img) / 255)
    img = rgb_to_hsv(rearrange(img, 'h w c -> 1 c h w'))
    img = rearrange(img.squeeze(0), 'c h w -> (h w) c')
    return img

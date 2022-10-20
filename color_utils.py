import logging
from io import BytesIO

from PIL import Image, ImageCms
from einops import rearrange
import numpy as np


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
        lab_icc = ImageCms.createProfile('LAB', colorTemp=6500)
        img = ImageCms.profileToProfile(image, orig_icc, lab_icc, outputMode='LAB')

    img = np.asarray(img)
    img = np.concatenate((img[..., 0:1], img.view(np.int8)[..., 1:3]), axis=-1)
    img = rearrange(img, 'h w c -> (h w) c')
    return img
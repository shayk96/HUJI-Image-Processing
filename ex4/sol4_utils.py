from scipy.signal import convolve2d
import numpy as np


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


"""---------------------------------imported functions from ex3------------------------------------------------------"""

import imageio as im
import skimage.color as ski
from scipy.ndimage.filters import convolve

GAUSSIAN_BASE = [1, 1]
MINIMUM_RESOLUTION = 16 * 2
HEIGHT = 0
WIDTH = 1
GRAYSCALE = 1
RGB = 2
MIN_VAL = 0
MAX_VAL_FLOAT = 1
T_FLOAT = np.float64


def normalize(img, new_min, new_max, data_type):
    """
  normalize elements in matrix to a given range
  :param img: image
  :param new_min: minimum value
  :param new_max: maximum value
  :param data_type: data type
  :return: image scaled to range (new_max - new_min)
  """
    old_range = (img.max() - img.min())
    new_range = (new_max - new_min)
    img = ((((img - img.min()) * new_range) / old_range) + new_min).astype(data_type)
    return img


def read_image(filename, representation):
    """
  Reads an image and converts it into a given representation
  :param filename: filename of image on disk
  :param representation: 1 for greyscale and 2 for RGB
  :return: Returns the image as a np.float64 matrix normalized to [0,1]
  """
    image = im.imread(filename)
    if representation == GRAYSCALE:
        image = ski.rgb2gray(image)
    image = normalize(image, MIN_VAL, MAX_VAL_FLOAT, T_FLOAT)
    return image


def reduce(im, blur_filter):
    """
  Reduces an image by a factor of 2 using the blur filter
  :param im: Original image
  :param blur_filter: Blur filter
  :return: the downsampled image
  """
    blurred_rows = convolve(im, blur_filter, mode="mirror")
    blurred_image = convolve(blurred_rows, blur_filter.T, mode="mirror")
    return blurred_image[::2, ::2]


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
  Builds a gaussian pyramid for a given image
  :param im: a grayscale image with double values in [0, 1]
  :param max_levels: the maximal number of levels in the resulting pyramid.
  :param filter_size: the size of the Gaussian filter
          (an odd scalar that represents a squared filter)
          to be used in constructing the pyramid filter
  :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
          standard python array with maximum length of max_levels,
          where each element of the array is a grayscale image.
          and filter_vec is a row vector of shape (1, filter_size)
          used for the pyramid construction.
  """
    filter_vec = create_blur_filter(filter_size)
    pyr = []
    lvl = 1
    res = im.shape
    pyr.append(im)
    while lvl < max_levels and (res[HEIGHT] >= MINIMUM_RESOLUTION and res[WIDTH] >= MINIMUM_RESOLUTION):
        image = reduce(im, filter_vec)
        pyr.append(image)
        im = image
        lvl += 1
        res = im.shape
    return pyr, filter_vec


def create_blur_filter(filter_size):
    """
  creates a gaussian vector that will be used to blur the images
  :param filter_size: the size of the vector to be created
  :return: a vector of shape(1,0), size of filter_size
  """
    filter = np.array(GAUSSIAN_BASE, dtype=np.float64)
    for i in range(filter_size - 2):
        filter = np.convolve(filter, GAUSSIAN_BASE)
    return np.array([filter / sum(filter)])




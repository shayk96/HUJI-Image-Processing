""" -----------------------------imports-----------------------------------------------------------------------------"""
import numpy as np
import imageio as im
import skimage.color as ski
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import os

""" -----------------------------constants---------------------------------------------------------------------------"""

GAUSSIAN_BASE = [1, 1]
MINIMUM_RESOLUTION = 16 * 2
HEIGHT = 0
WIDTH = 1
SUBPLOT_IND = 2
CHANNELS = 3
MASK = "mask"

""" -----------------------------helper functions--------------------------------------------------------------------"""


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


def create_plots(im1, im2, im_blend, mask, title1, title2, title_blend):
    fig, ((ax_im1, ax_im2), (ax_mask, ax_blend)) = plt.subplots(SUBPLOT_IND, SUBPLOT_IND)

    ax_im1.imshow(im1)
    ax_im2.imshow(im2)
    ax_mask.imshow(mask, cmap='gray')
    ax_blend.imshow(im_blend)

    ax_mask.set_title(MASK)
    ax_blend.set_title(title_blend)
    ax_im2.set_title(title2)
    ax_im1.set_title(title1)

    ax_blend.set_axis_off()
    ax_mask.set_axis_off()
    ax_im2.set_axis_off()
    ax_im1.set_axis_off()

    fig.show()


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


""" -----------------------------required functions------------------------------------------------------------------"""


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


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    filter = blur_filter * 2
    im = np.insert(im, np.arange(1, im.shape[1] + 1), 0, axis=1)
    im = np.insert(im, np.arange(1, im.shape[0] + 1), 0, axis=0)
    blurred_rows = convolve(im, filter, mode="mirror")
    blurred_image = convolve(blurred_rows, filter.T, mode="mirror")
    return blurred_image


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


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    gaussian_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lapl_pyr = []
    for lvl in range(len(gaussian_pyr) - 1):
        lapl_im = gaussian_pyr[lvl] - expand(gaussian_pyr[lvl + 1], filter_vec)
        lapl_pyr.append(lapl_im)
    lapl_pyr.append(gaussian_pyr[-1])
    return lapl_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    for i in range(len(coeff)):
        lpyr[i] *= coeff[i]
    i = len(lpyr) - 2
    img = lpyr[-1]
    while i >= 0:
        img = lpyr[i] + expand(img, filter_vec)
        i -= 1
    return img


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    for lvl in range(levels):
        if lvl == 0:
            res = normalize(pyr[0], 0, 1, np.float64)
            continue
        img = normalize(pyr[lvl], 0, 1, np.float64)
        diff = res.shape[0] - img.shape[0]
        img = np.pad(img, ((0, diff), (0, 0)), mode='constant', constant_values=0)
        res = np.concatenate((res, img), 1)
    return res


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.axis('off')
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    mask = mask.astype('float64')
    mask_gaus, vec = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    im1_lapl, vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_lapl, vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    lvls = len(im1_lapl)
    blanded_lapl = [0] * lvls
    for lvl in range(lvls):
        blanded_lapl[lvl] = mask_gaus[lvl] * im1_lapl[lvl] + (1 - mask_gaus[lvl]) * im2_lapl[lvl]
    im_blend = laplacian_to_image(blanded_lapl, vec, [1] * len(blanded_lapl))
    return normalize(im_blend, 0, 1, np.float64)


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('images_to_blend/hot_dog1.jpg'), RGB)
    im2 = read_image(relpath('images_to_blend/dog1.jpg'), RGB)
    mask = read_image(relpath('images_to_blend/mask1.jpg'), GRAYSCALE).round().astype(np.bool)
    im_blend = np.zeros(im1.shape)
    for channel in range(CHANNELS):
        im_blend[:, :, channel] = pyramid_blending(im1[:, :, channel], im2[:, :, channel], mask, 5, 9, 9)
    create_plots(im1, im2, im_blend, mask, "hot dog", "dog", "hot dog dog")
    return im1, im2, mask, im_blend


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('images_to_blend/staff2.jpg'), RGB)
    im2 = read_image(relpath('images_to_blend/harry_potter2.jpg'), RGB)
    mask = read_image(relpath('images_to_blend/mask2.jpg'), GRAYSCALE).round().astype(np.bool)
    im_blend = np.zeros(im1.shape)
    for channel in range(CHANNELS):
        im_blend[:, :, channel] = pyramid_blending(im1[:, :, channel], im2[:, :, channel], mask, 5, 9, 9)
    create_plots(im1, im2, im_blend, mask, "staff", "hary potter", "potter staff")
    return im1, im2, mask, im_blend


""" -----------------------------imported functions from ex1---------------------------------------------------------"""

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
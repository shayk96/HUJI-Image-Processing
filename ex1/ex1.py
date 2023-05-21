""" ---------------------------------- imports -------------------------------------------------------------------"""
import numpy as np
import skimage.color as ski
import imageio as im
import matplotlib.pyplot as plt

""" ---------------------------------- constants -----------------------------------------------------------------"""

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
RGB_YIQ_TRANSFORMATION_MATRIX_INVERSE_T = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX).transpose()
RANGE = 255
RGB_DIMENSIONS = 3
Y_CHANEL = 0
BINS = 256
MIN_VAL = 0
MAX_VAL_INT = 255
MAX_VAL_FLOAT = 1
T_FLOAT = np.float64
T_INT = np.uint8
R_CHANNEL = 0
G_CHANNEL = 1
B_CHANNEL = 2

""" ---------------------------------- additional functions ------------------------------------------------------"""


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


def extract_chanel(im_orig):
    """
    extracts the channel to morph
    :param im_orig: the original image
    :return: image : the channel to work on, encase of GRAYSCALE the whole image
             rgb_image : a boolean indicating if the original image is RGB or GRAYSCALE
             yiq_img: if im_orig is RGB than a YIQ representation of the image else NONE
    """
    if np.ndim(im_orig[:, :, ]) == RGB_DIMENSIONS:
        yiq_img = rgb2yiq(im_orig)
        image = normalize(yiq_img[:, :, Y_CHANEL], MIN_VAL, MAX_VAL_INT, T_INT)
        rgb_img = True
    else:
        yiq_img = None
        image = normalize(im_orig, MIN_VAL, MAX_VAL_INT, T_INT)
        rgb_img = False
    return image, rgb_img, yiq_img


def calc_hist(image):
    """
    calculates the different histograms for the original image if GREYSCALE or for the Y channel if RGB
    :param image: the original image if GREYSCALE else Y channel only
    :return: hist_o - original histogram
             hist_t - normalized and cumulative histogram
             bounds_o - bounds of the histogram
    """
    hist_o, bounds_o = np.histogram(image, bins=BINS)
    hist_c = hist_o.cumsum()
    min_v = hist_c[hist_c > MIN_VAL].min()
    hist_t = (RANGE * (hist_c - min_v) / (hist_c[RANGE] - min_v)).astype(T_INT)
    return bounds_o, hist_o, hist_t


def calculate_optimal_z_q(errors, hist_o, n_iter, n_quant, q, z, z_temp):
    """
    calculates the optimal boundaries(z) and optimal value in each quant(q) while computing the total error in each
    iteration
    :param errors: a list containing the error from each iteration
    :param hist_o: the original histogram of the image or channel
    :param n_iter: max number of iterations to compute while searching for optimal solution
    :param n_quant: the number of quants to divide the image to
    :param q: a list containing the optimal value in each quant
    :param z: the boundaries
    :param z_temp: each iteration previous boundaries to compare to in order to check for convergence
    :return: a list of all the errors
    """
    for i in range(n_iter):
        error = 0
        for g in range(n_quant):
            grey_levels = np.array(range(z[g] + 1, z[g + 1] + 1))
            pixels_per_level = np.array(hist_o[z[g] + 1:z[g + 1] + 1]).T
            q[g] = np.dot(grey_levels, pixels_per_level) / np.sum(pixels_per_level)
            error += np.dot(np.power((grey_levels - q[g]), 2), pixels_per_level)
        for j in range(1, len(z) - 1):
            z[j] = (q[j - 1] + q[j]) / 2
        if (z == z_temp).all():
            break
        z_temp = np.copy(z)
        errors = np.append(errors, error)
    return errors


""" ---------------------------------- required functions---------------------------------------------------------"""


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


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    im_yiq = np.dot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.transpose())
    return im_yiq


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
           the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    im_rgb = np.dot(imYIQ, RGB_YIQ_TRANSFORMATION_MATRIX_INVERSE_T)
    return im_rgb


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    image, rgb_img, yiq_img = extract_chanel(im_orig)
    bounds_o, hist_o, hist_t = calc_hist(image)
    if rgb_img:
        yiq_img[:, :, Y_CHANEL] = normalize(np.interp(image, bounds_o[:-1], hist_t), MIN_VAL, MAX_VAL_FLOAT, T_FLOAT)
        image_eq = yiq2rgb(yiq_img)
    else:
        image_eq = normalize(np.interp(image, bounds_o[:-1], hist_t), MIN_VAL, MAX_VAL_FLOAT, T_FLOAT)
    hist_eq, bounds_eq = np.histogram(image_eq, bins=BINS)
    return image_eq, hist_o, hist_eq


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    image, rgb_img, yiq_img = extract_chanel(im_orig)
    errors, q, z = calculate_quantization(image, n_iter, n_quant)
    if rgb_img:
        image_quant = np.copy(yiq_img)
    else:
        image_quant = np.zeros(np.shape(im_orig))
    for i in range(n_quant):
        if rgb_img:
            image_quant[:, :, Y_CHANEL] = np.where((z[i] + 1 <= image) & (image <= z[i + 1]), q[i],
                                                   image_quant[:, :, Y_CHANEL])
        else:
            image_quant = np.where((z[i] + 1 <= image) & (image <= z[i + 1]), q[i], image_quant)
    if rgb_img:
        image_quant[:, :, Y_CHANEL] = normalize(image_quant[:, :, Y_CHANEL], MIN_VAL, MAX_VAL_FLOAT, T_FLOAT)
        image_quant = yiq2rgb(image_quant)
    else:
        image_quant = normalize(image_quant, MIN_VAL, MAX_VAL_FLOAT, T_FLOAT)
    return image_quant, errors


def calculate_quantization(image, n_iter, n_quant):
    """
`   calculates the optimal q and z for the given image
    :param image: Input float64 [0,255] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return: q - the optimal values to which each of the segments’ intensities will map.
             z - the borders which divide the histograms into segments.
              errors - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    hist_o, bound_o = np.histogram(image, bins=BINS)
    hist_c = hist_o.cumsum()
    pixels_per_block = hist_c[-1] / n_quant
    z = np.zeros(n_quant + 1, dtype=np.int64)
    for i in range(n_quant):
        z[i] = np.argmax((pixels_per_block * i <= hist_c) & (hist_c <= pixels_per_block * (i + 1)))
    z[-1] = RANGE
    z[0] = -1
    z_temp = np.copy(z)
    errors = np.array([])
    q = np.zeros(n_quant, dtype=np.int64)
    errors = calculate_optimal_z_q(errors, hist_o, n_iter, n_quant, q, z, z_temp)
    return errors, q, z


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    from scipy.cluster.vq import kmeans2
    dimension = (im_orig.shape[R_CHANNEL] * im_orig.shape[G_CHANNEL], im_orig.shape[B_CHANNEL])
    im_flatt = np.reshape(im_orig, dimension)
    center, label = kmeans2(im_flatt, n_quant)
    im_quant = np.reshape(center[label], im_orig.shape)
    return im_quant
""" -----------------------------imports-----------------------------------------------------------------------------"""
import numpy as np
import imageio as im
import scipy.io.wavfile as wavfile
from scipy.signal import convolve2d
import skimage.color as ski

""" -----------------------------constants---------------------------------------------------------------------------"""
MINUS = -1
PLUS = 1
RATE = 0
DATA = 1
EXP_VAR = 2j * np.pi
OUTPUT_FILE_BY_RATE = "change_rate.wav"
OUTPUT_FILE_BY_SAMPLES = "change_samples.wav"
NORMAL_SPEED = 1
CONV = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
CONV_2D_MODE = "same"
CONV_2D_BOUNDARY = "symm"
ROWS = 0
COLS = 1

""" -----------------------------helper functions--------------------------------------------------------------------"""


def DFT_IDFT_helper(signal, sign):
    """
    the function calculates the 1D fourier representation of the signal or the original representation if the signal is
    in fourier representation
    :param signal: the signal to transform
    :param sign: the sign of the power in the exponent
    :return: the transformed signal
    """
    signal_len = np.shape(signal)[ROWS]
    omega = ((sign * EXP_VAR) / signal_len)
    power_matrix = np.fromfunction(lambda i, j: np.exp(omega * i * j), (signal_len, signal_len))
    result = power_matrix @ signal.reshape(-1, 1)
    if sign == PLUS:
        result *= (1 / signal_len)
    return result.reshape(np.shape(signal))


def DFT2_IDFT2_helper(signal, function):
    """
    the function calculates the 2D fourier representation of the signal or the original representation if the signal is
    in fourier representation
    :param signal: the signal to transform
    :param function: the function to run on the rows and columns of the image. can be DFT or IDFT
    :return: the transformed signal
    """
    fourier_rows = np.apply_along_axis(function, ROWS, signal)
    fourier_cols = np.apply_along_axis(function, COLS, fourier_rows)
    return fourier_cols


def calculate_padding(data, new_num_samples):
    """
    adds zeros to the data in order to pad it. in case of odd number adds more to the right of the data
    :param data: the data
    :param new_num_samples:
    :return:
    """
    padding_len = new_num_samples - len(data)
    padding_left = np.zeros(int(padding_len / 2))
    padding_right = np.zeros(int(padding_len / 2 + (padding_len % 2)))
    return padding_left, padding_right


def calc_magnitude(dx, dy, dtype):
    """
    calculates the magnitude of an image using the derivative if x and the derivative of y
    :param dx: the derivative in the x scale
    :param dy: the derivative in the y scale
    :param dtype: the data type to return the result
    :return: the magnitude of the image derivative
    """
    magnitude = (np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)).astype(dtype)
    return magnitude


""" -----------------------------required functions------------------------------------------------------------------"""


def DFT(signal):
    """
    function that transform a 1D discrete signal to its Fourier representation
    :param signal: array of data type float64 with shape (N,) or (N,1)
    :return: the complex Fourier signal, the same shape as the input
    """
    return DFT_IDFT_helper(signal, MINUS)


def IDFT(fourier_signal):
    """
    function that transform a 1D discrete signal to its Fourier representation
    :param fourier_signal: array of data type complex128 with shape (N,) or (N,1)
    :return: the complex signal, the same shape as the input
    """
    return DFT_IDFT_helper(fourier_signal, PLUS)


def DFT2(image):
    """
    function that transform a 2D discrete signal to its Fourier representation
    :param image: : image is a grayscale image of data type float64
    :return: the complex Fourier signal, the same shape as the input
    """
    return DFT2_IDFT2_helper(image, DFT)


def IDFT2(fourier_image):
    """
    function that transform a 1D discrete signal to its Fourier representation
    :param fourier_image: array of data type complex128 with shape (N,) or (N,1)
    :return: the complex signal, the same shape as the input
    """
    return DFT2_IDFT2_helper(fourier_image, IDFT)


def change_rate(filename, ratio):
    """
    reads an audio file, changes the ratio of the samples and saves it to the OUTPUT_FILE
    :param filename:  a string representing the path to a WAV file
    :param ratio:  a positive float64 representing the duration change
    """
    sample_rate, data = wavfile.read(filename)
    new_ratio = int((sample_rate * ratio))
    wavfile.write(OUTPUT_FILE_BY_RATE, new_ratio, data)


def change_samples(filename, ratio):
    """
    function that changes the duration of an audio file by reducing the number of samples using Fourier
    :param filename:  a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    :return: a 1D ndarray of data type float64 representing the new sample points.
    """
    sample_rate, data = wavfile.read(filename)
    new_data = resize(data, ratio)
    wavfile.write(OUTPUT_FILE_BY_SAMPLES, sample_rate, new_data)
    return new_data


def resize(data, ratio):
    """
    resizes the number of samples to the correct ratio
    :param data: 1D ndarray of data type float64 or complex128 representing the original sample points
    :param ratio: a positive float64 representing the duration change.
    :return: a 1D ndarray of the data type of data representing the new sample points
    """
    if ratio == NORMAL_SPEED:
        return data
    new_num_samples = np.round((len(data) / ratio))
    if ratio < NORMAL_SPEED:
        padding_left, padding_right = calculate_padding(data, new_num_samples)
    fourier_data = np.fft.fftshift(DFT(data))
    if ratio < NORMAL_SPEED:
        fourier_data = np.concatenate([padding_left, fourier_data, padding_right])
    else:
        fourier_data = fourier_data[int(len(data) / 2 - new_num_samples / 2): int(len(data) / 2 + new_num_samples / 2)]
    fourier_data = np.fft.ifftshift(fourier_data)
    new_data = (IDFT(fourier_data)).round().astype(data.dtype)
    return np.real(new_data)


def resize_spectrogram(data, ratio):
    """
    a function that speeds up a WAV file, without changing the pitch, using spectrogram scaling
    :param data: a 1D ndarray of data type float64 representing the original sample points
    :param ratio: a positive float64 representing the rate change of the WAV file
    :return: the new sample points according to ratio with the same datatype as data.
    """
    spectro = stft(data)
    changed_spectro = np.apply_along_axis(lambda row: resize(row, ratio), 1, spectro)
    changed_data = istft(changed_spectro).round().astype(data.dtype)
    return changed_data


def resize_vocoder(data, ratio):
    """
    a function that speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio:  a positive float64 representing the rate change of the WAV file
    :return: n the given data rescaled according to ratio with the same datatype as data.
    """
    spectro = stft(data)
    wrapped_spectro = phase_vocoder(spectro, ratio)
    changed_data = istft(wrapped_spectro).astype(data.dtype)
    return changed_data


def conv_der(im):
    """
     a function that computes the magnitude of image derivatives in image space
    :param im: image to derive
    :return:the magnitude of the derivative, with the same dtype and shape as im
    """
    dx = convolve2d(im, CONV, mode=CONV_2D_MODE, boundary=CONV_2D_BOUNDARY)
    dy = convolve2d(im, CONV.T, mode=CONV_2D_MODE, boundary=CONV_2D_BOUNDARY)
    return calc_magnitude(dx, dy, im.dtype)


def fourier_der(im):
    """
    a function that computes the magnitude of image derivatives in fourier space
    :param im: image to derive
    :return:the magnitude of the derivative, with the same dtype and shape as im
    """
    size_x, size_y = np.shape(im)
    im_fourier = np.fft.fftshift(DFT2(im))
    u, v = np.meshgrid(np.arange(-size_y // 2, size_y // 2), np.arange(-size_x // 2, size_x // 2))
    dx = np.real((EXP_VAR / size_x) * IDFT2(np.fft.ifftshift(im_fourier * u)))
    dy = np.real((EXP_VAR / size_y) * IDFT2(np.fft.ifftshift(im_fourier * v)))
    return calc_magnitude(dx, dy, im.dtype)


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


"""--------------------------ex2 helper------------------------------------------------------------------------------"""
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
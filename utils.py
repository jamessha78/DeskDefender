import multiprocessing
import scipy.signal
import numpy as np

from PIL import Image
import sys
import time
import math


DEFAULT_BINS = 10
NUM_THREADS = 8
# USE_THREADING = True
USE_THREADING = False


def parallelize(function, args, should_map_over_arg, pool=None):
    """
    A helper function to parallelize calling a function. Only runs in parallel if the USE_THREADING constant is set
    to True.
    @param function: The function to be mapped over
    @param args: A list of the args to pass to the function. If an arg changes for each iteration, it should be a list
    itself, otherwise if it is the same for each map iteration, it can be any arbitrary value
    @param should_map_over_arg: A list the same length as args containing booleans indicating whether or not the
    corresponding argument changes over each iteration.
    @param pool: The processor pool to run the operation over. If omitted a processor pool will be created.
    @return: An array containing the result of mapping the function over the args
    """
    num_args_list = [len(arg) for i, arg in enumerate(args) if should_map_over_arg[i]]
    num_iterations = num_args_list[0]
    if not all(map(lambda x: x == num_iterations, num_args_list)):
        raise ValueError("Not all changing args are the lists of the same length")

    make_own_pool = pool is None and USE_THREADING
    if make_own_pool:
        pool = multiprocessing.Pool(NUM_THREADS)

    def get_arg(arg_index, map_iter):
        if should_map_over_arg[arg_index]:
            return args[arg_index][map_iter]
        else:
            return args[arg_index]

    def args_generator():
        for map_iteration in range(num_iterations):
            yield [get_arg(arg_index, map_iteration) for arg_index in range(len(args))]

    if USE_THREADING:
        chunksize = len(args) // NUM_THREADS + 1
        rtn = pool.map_async(function, args_generator(), chunksize=chunksize).get(99999)
    else:
        rtn = map(function, args_generator())

    if make_own_pool:
        pool.close()

    return rtn


def extract_features(image, cell_size, window_size, pool=None):
    """
    @param image: A 2d array representing an image
    @param cell_size: the size of a single cell, must be even
    @return: A feature matrix of shape [num_windows, num_features]
    """
# def __extract_row(image, cell_size, window_size):
    image = preprocess_image(image, cell_size)
    h, w, d = image.shape
    rows = h // cell_size
    cols = w // cell_size
    grouped = np.vstack(parallelize(
        __extract_row,
        [range(rows), cols, cell_size, image, d],
        [True, False, False, False, False],
        pool
    ))
    #log_since("Grouped initial image", start_time)

    rows, cols, d = grouped.shape

    window_height = window_size[0] // cell_size
    window_width = window_size[1] // cell_size
    num_cells_wide = cols - (window_width - 1)
    num_cells_tall = rows - (window_height - 1)

    num_windows = num_cells_wide * num_cells_tall
    num_features_per_window = window_height * window_width * d
    features = np.empty((num_windows, num_features_per_window))
    positions = np.empty((num_windows, 4))
    assert isinstance(features, np.ndarray)

    cur_win = 0
    #log_since("Sliding window", start_time)
    for i in xrange(num_cells_tall):
        for j in xrange(num_cells_wide):
            features[cur_win, :] = grouped[i:i+window_height, j:j+window_width, :].flatten()

            positions[cur_win, :] = np.array(
                [i*cell_size, j*cell_size, (i+window_height)*cell_size, (j+window_width)*cell_size])
            cur_win += 1
    #log_since("Done with feature extraction", start_time)
    return features, positions


def __extract_row(args):
    row, cols, cell_size, image, d = args
    extracted = np.empty((1, cols, d))
    for c in xrange(cols):
        start_row = row * cell_size
        end_row = (row + 1) * cell_size
        start_col = c * cell_size
        end_col = (c + 1) * cell_size
        extracted[0, c, :] = normalize(image[start_row:end_row, start_col:end_col, :].sum((0, 1)))
    return extracted


def preprocess_image(image, cell_size, num_bins=DEFAULT_BINS):
    """
    Converts the image into a format that is easy to get the hog of.
    Basically precomutes which and the magnitude of the contribution for each pixel ahead of time.
    @param image: An n x m matrix of pixel values
    @type image: np.ndarray
    @return: An n x m x b matrix, where b is the number of bins and each n x m matrix contains contributions to the
    corresponding bin.
    @rtype: np.ndarray
    """
    # Normalize the patches within the image
    normalized_image = np.zeros(image.shape)
    assert isinstance(normalized_image, np.ndarray)
    h, w = image.shape
    pre_convolution_cell_size = cell_size
    x_range = w // pre_convolution_cell_size
    y_range = h // pre_convolution_cell_size
    for x in range(x_range):
        start_x = x * pre_convolution_cell_size
        for y in range(y_range):
            start_y = y * pre_convolution_cell_size
            if x_range - 1 == x:
                end_x = w
            else:
                end_x = start_x + pre_convolution_cell_size
            if y_range - 1 == y:
                end_y = h
            else:
                end_y = start_y + pre_convolution_cell_size

            normalized_image[start_y:end_y, start_x:end_x] = normalize(image[start_y:end_y, start_x:end_x])
    image = normalized_image

    angles, mags = get_mags_angles(image)
    bins = np.linspace(-np.pi, np.pi, num_bins+1, endpoint=True)
    bins[-1] += .01  # In case something is exactly the upper bound, want to to catch it with a strict inequality
    output = np.empty(angles.shape + (num_bins,))
    assert isinstance(output, np.ndarray)
    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        output[:, :, i] = np.logical_and(lower_bound <= angles, angles < upper_bound) * mags
    return output


def get_mags_angles(image):
    image = image.astype(np.float32)

    # Faster way to compute the tap derivative
    x_deriv = image[1:-1, :-2] - image[1:-1, 2:]
    y_deriv = image[:-2, 1:-1] - image[2:, 1:-1]
    angles = np.arctan2(y_deriv, x_deriv)
    mags = np.sqrt(np.square(x_deriv) + np.square(y_deriv))
    return angles, mags


def normalize(array):
    norm = np.linalg.norm(array.flatten(), 2)
    if norm != 0:
        return array / norm
    else:
        return array


def log(msg, permanent=True):
    time_stamp = time.asctime()
    line_ending = "\n" if permanent else "\r"
    template = "[" + time_stamp + "] %s" + " " * 10 + line_ending
    sys.stdout.write(template % msg)
    sys.stdout.flush()


def log_since(msg, start_time):
    elapsed_time = time.time() - start_time
    print "[%.10f] %s" % (elapsed_time, msg)

import multiprocessing
import scipy.signal
import numpy as np

from PIL import Image
import sys
import time
import math


DEFAULT_BINS = 10
NUM_THREADS = 8
USE_THREADING = True
#USE_THREADING = False


def do_grouping(image, cell_size, x_offset=0, y_offset=0):
    h, w, d = image.shape
    rows = (h - y_offset) // cell_size
    cols = (w - x_offset) // cell_size
    grouped = np.empty((rows, cols, d))
    assert isinstance(grouped, np.ndarray)
    for r in xrange(rows):
        for c in xrange(cols):
            start_row = r * cell_size + y_offset
            end_row = (r + 1) * cell_size + y_offset
            start_col = c * cell_size + x_offset
            end_col = (c + 1) * cell_size + x_offset
            grouped[r, c] = image[start_row:end_row, start_col:end_col, :].sum((0, 1))
    return grouped


def num_groups(height, width, stride, x_offset, y_offset):
    rows = (height - y_offset) // stride
    cols = (width - x_offset) // stride
    return rows * cols


def get_feature_matrix_from_image(image, block_size=6, window_size=(38, 30)):
    """
    @param image: A 2d array representing an image
    @param block_size: the size of a single block, must be even
    @return: A feature matrix of shape [num_windows, num_features]
    """
    #start_time = time.time()
    #if block_size % 2 != 0:
    #    raise ValueError("Block size must be even")
    #cell_size = block_size / 2
    cell_size = block_size
    image = preprocess_image(image)
    grouped = do_grouping(image, cell_size)
    #log_since("Grouped initial image", start_time)

    rows, cols, d = grouped.shape

    window_height = window_size[0] // cell_size
    window_width = window_size[1] // cell_size
    num_cells_wide = cols/2 - (window_width/2 - 1)
    num_cells_tall = rows/2 - (window_height/2 - 1)

    #aligned_size = num_groups(window_height, window_width, 2, 0, 0) * d
    #vert_offset_size = num_groups(window_height, window_width, 2, 0, 1) * d
    #horiz_offset_size = num_groups(window_height, window_width, 2, 1, 0) * d
    #both_offset_size = num_groups(window_height, window_width, 2, 1, 1) * d

    num_windows = num_cells_wide * num_cells_tall
    #num_features_per_window = aligned_size + vert_offset_size + horiz_offset_size + both_offset_size
    num_features_per_window = window_height * window_width * d
    features = np.empty((num_windows, num_features_per_window))
    positions = np.empty((num_windows, 4))
    assert isinstance(features, np.ndarray)

    cur_win = 0
    #log_since("Sliding window", start_time)
    for i in xrange(num_cells_tall):
        for j in xrange(num_cells_wide):
            features[cur_win, :] = grouped[i:i+window_height, j:j+window_width, :].flatten()
            #window = grouped[i:i+window_height, j:j+window_width, :]
            #cur_feat = 0
            #features[cur_win, cur_feat:cur_feat+aligned_size] = do_grouping(window, 2, 0, 0).flatten()
            #cur_feat += aligned_size
            #features[cur_win, cur_feat:cur_feat+vert_offset_size] = do_grouping(window, 2, 0, 1).flatten()
            #cur_feat += vert_offset_size
            #features[cur_win, cur_feat:cur_feat+horiz_offset_size] = do_grouping(window, 2, 1, 0).flatten()
            #cur_feat += horiz_offset_size
            #features[cur_win, cur_feat:cur_feat+both_offset_size] = do_grouping(window, 2, 1, 1).flatten()

            positions[cur_win, :] = np.array(
                [i*cell_size, j*cell_size, (i+window_height)*cell_size, (j+window_width)*cell_size])
            cur_win += 1
    #log_since("Done with feature extraction", start_time)
    return features, positions


def get_mags_angles(image):
    image = image.astype(np.float32)

    # Faster way to compute the tap derivative
    x_deriv = image[1:-1, :-2] - image[1:-1, 2:]
    y_deriv = image[:-2, 1:-1] - image[2:, 1:-1]
    angles = np.arctan2(y_deriv, x_deriv)
    mags = np.sqrt(np.square(x_deriv) + np.square(y_deriv))
    return angles, mags


def get_hog(patch_angles, patch_mags, bins=10):
    hist, bin_edges = np.histogram(patch_angles, bins, range=(-np.pi, np.pi), weights=patch_mags, density=True)
    return np.nan_to_num(hist)


def extract_hog_features(patch_dicts, images, patch_extractor, num_bins=DEFAULT_BINS, pool=None):
    """
    Creates a feature matrix of the shape [num_patches, num_features]. Is multi-threaded for performance
    @param patch_dicts: The input patch dictionaries.
    @type patch_dicts: list[dict]
    @param images: The list of downsampled images
    @type images: list[ndarray]
    @param patch_extractor: The patch extractor
    @type patch_extractor: patch_extractor.PatchExtractor
    @param pool: A  thread pool used to extract features in parallel
    @type pool: multiprocessing.Pool
    @return: Matrix of features
    @rtype: ndarray
    """
    def data_iterator():
        for patch in patch_dicts:
            yield patch, images, patch_extractor, num_bins
    make_own_pool = pool is None and USE_THREADING
    if make_own_pool:
        pool = multiprocessing.Pool(NUM_THREADS)
    if USE_THREADING:
        features = pool.map_async(__extraction_helper, data_iterator()).get(99999)
    else:
        features = map(__extraction_helper, data_iterator())
    features = np.array(features)

    if make_own_pool:
        pool.close()

    return features


def __extraction_helper(args):
    # Do ugly unpacking to allow this to be called by map for multi-threading
    patch, images, patch_extractor, num_bins = args
    positions = patch_extractor.patch_positions_from_position(patch['patch_position'], patch['patch_size'])
    sub_patch_size = np.array(patch_extractor.rf_size)
    num_patches = positions.shape[1]
    image = images[patch['image_index']]
    #sub_patches = patch_extractor.extract_all(patch)

    # The length of each sub-patch should equal num_bins. Not doing error checking to save time
    feature_vec = np.empty(num_patches * num_bins)
    for i in xrange(num_patches):
        start = positions[:, i]
        end = start + sub_patch_size
        feature_vec[i*num_bins:(i+1)*num_bins] = image[start[0]:end[0], start[1]:end[1], :].sum((0, 1))
    return feature_vec


def preprocess_image(image, num_bins=DEFAULT_BINS):
    """
    Converts the image into a format that is easy to get the hog of.
    Basically precomutes which and the magnitude of the contribution for each pixel ahead of time.
    @param image: An n x m matrix of pixel values
    @type image: np.ndarray
    @return: An n x m x b matrix, where b is the number of bins and each n x m matrix contains contributions to the
    corresponding bin.
    @rtype: np.ndarray
    """
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


def get_integral_image(image):
    s = np.zeros(image.shape)
    ii = np.zeros(image.shape)
    s[0, :] = image[0, :]
    for i in range(1, image.shape[0]):
        s[i, :] = s[i-1, :] + image[i, :]
    ii[:, 0] = s[:, 0]
    for j in range(1, image.shape[1]):
        ii[:, j] = ii[:, j-1] + s[:, j]
    return ii


def log(msg, permanent=True):
    time_stamp = time.asctime()
    line_ending = "\n" if permanent else "\r"
    template = "[" + time_stamp + "] %s" + " " * 10 + line_ending
    sys.stdout.write(template % msg)
    sys.stdout.flush()


def log_since(msg, start_time):
    elapsed_time = time.time() - start_time
    print "[%.10f] %s" % (elapsed_time, msg)

#image = Image.open('images/newtest/ew-friends.gif').convert('L')
#image = np.array(image)
#print image[:200, :200]
#angles, mags = get_mags_angles(image)
#hog = get_hog(angles, mags)
#print hog

#Image.fromarray(mags).show()
#Image.fromarray(image).show()

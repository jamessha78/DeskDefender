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


def get_mags_angles(image):
    image = image.astype(np.float32)
    #tap = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    #x_deriv = scipy.signal.convolve2d(image, tap, mode='valid')
    #y_deriv = scipy.signal.convolve2d(image, tap.T, mode='valid')

    # Faster way to compute the tap derivative
    x_deriv = image[1:-1, :-2] - image[1:-1, 2:]
    y_deriv = image[:-2, 1:-1] - image[2:, 1:-1]
    angles = np.arctan2(y_deriv, x_deriv)
    mags = np.sqrt(np.square(x_deriv) + np.square(y_deriv))
    return angles, mags


def get_hog(patch_angles, patch_mags, bins=10):
    hist, bin_edges = np.histogram(patch_angles, bins, range=(-np.pi, np.pi), weights=patch_mags, density=True)
    return np.nan_to_num(hist)


def extract_hog_features(patches, patch_extractor, num_bins=DEFAULT_BINS, pool=None):
    """
    Creates a feature matrix of the shape [num_patches, num_features]. Is multi-threaded for performance
    @param patches: The input patches.
    #angles, mags = get_mags_angles(patch)
    #angle_patches = patch_extractor.extract_all(angles)
    #mag_patches = patch_extractor.extract_all(mags)
    #feature_vec = np.zeros((angle_patches.shape[0], num_bins))
    #patch_size = patch_extractor.rf_size
    #for i in range(angle_patches.shape[0]):
    #    angle_patch = angle_patches[i, :].reshape(patch_size[0], patch_size[1])
    #    mag_patch = mag_patches[i, :].reshape(patch_size[0], patch_size[1])
    #    hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
    #    feature_vec[i, :] = hog_features

    #return feature_vec.flatten() Each patch should be in the format returned by utils.preprocess_image
    @type patches: iterable[preprocessed_image]
    @param patch_extractor: The patch extractor
    @type patch_extractor: patch_extractor.PatchExtractor
    @param pool: A  thread pool used to extract features in parallel
    @type pool: multiprocessing.Pool
    @return: Matrix of features
    @rtype: ndarray
    """
    def data_iterator():
        for patch in patches:
            yield patch, patch_extractor, num_bins
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
    patch, patch_extractor, num_bins = args
    sub_patches = patch_extractor.extract_all(patch)

    # The length of each sub-patch should equal num_bins. Not doing error checking to save time
    feature_vec = np.empty(len(sub_patches) * num_bins)
    for i, sub_patch in enumerate(sub_patches):
        feature_vec[i*num_bins:(i+1)*num_bins] = np.sum(np.sum(sub_patch, 0), 0).flatten()
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

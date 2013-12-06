import scipy.signal
import numpy as np

from PIL import Image

def get_mags_angles(image):
    tap = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    x_deriv = scipy.signal.convolve2d(image, tap, mode='valid')
    y_deriv = scipy.signal.convolve2d(image, tap.T, mode='valid')
    angles = np.arctan2(y_deriv, x_deriv)
    mags = np.sqrt(np.square(x_deriv) + np.square(y_deriv))
    return angles, mags

def get_hog(patch_angles, patch_mags, bins=10):
    angle_block = 2*np.pi/(bins-1)
    hog = np.zeros([1, bins])
    size = patch_mags.shape
    for i in range(size[0]):
        for j in range(size[1]):
            mag = patch_mags[i, j]
            angle = patch_angles[i, j]
            if angle < 0:
                 angle += 2*np.pi
            bin_low = angle//angle_block
            bin_high = bin_low + 1
            low_weight = 1 - (patch_angles[i, j] % bins)/bins
            high_weight = 1 - low_weight
            hog[0, bin_low] += mag*low_weight
            hog[0, bin_high] += mag*high_weight
    hog = hog/max(np.sum(hog), 1e-5)
    return hog

def extract_hog_features(patch_extractor, patch, num_bins=10):
    angles, mags = get_mags_angles(patch)
    angle_patches = patch_extractor.extract_all(angles)
    mag_patches = patch_extractor.extract_all(mags)
    feature_vec = np.zeros((angle_patches.shape[0], num_bins))
    for i in range(angle_patches.shape[0]):
        angle_patch = angle_patches[i, :].reshape(patch_extractor.rf_size[0], patch_extractor.rf_size[1])
        mag_patch = mag_patches[i, :].reshape(patch_size, patch_size)
        hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
        feature_vec[i, :] = hog_features

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


#image = Image.open('images/newtest/ew-friends.gif').convert('L')
#image = np.array(image)
#print image[:200, :200]
#angles, mags = get_mags_angles(image)
#hog = get_hog(angles, mags)
#print hog

#Image.fromarray(mags).show()
#Image.fromarray(image).show()

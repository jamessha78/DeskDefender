import scipy.signal
import numpy as np

from PIL import Image
from sklearn import svm

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
    hog = hog/np.sum(hog)
    return hog

image = Image.open('images/newtest/ew-friends.gif').convert('L')
image = np.array(image)
angles, mags = get_mags_angles(image)
hog = get_hog(angles, mags)
print hog

#Image.fromarray(mags).show()

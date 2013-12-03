import pickle
import glob
import time

from sklearn import svm
from PIL import Image

from patch_extractor import *
from utils import *

f = open('classifier_svm.pickle', 'r')
clf = pickle.load(f)
f.close()

base_dir = 'cropped_images/newtest/'
img_paths = glob.glob(base_dir + '*.bmp')

num_correct = 0
for img_path in img_paths:
    img = Image.open(img_path).convert('L')
    num_bins = 10
    patch_size = 5
    patch_extractor = PatchExtractor(patch_size, 1, stride=patch_size/2)

    angles, mags = get_mags_angles(img)
    s = time.time()
    angle_patches = patch_extractor.extract_all(angles)
    mag_patches = patch_extractor.extract_all(mags)
    feature_vec = np.zeros((angle_patches.shape[0], num_bins))
    for i in range(angle_patches.shape[0]):
        angle_patch = angle_patches[i, :].reshape(patch_size, patch_size)
        mag_patch = mag_patches[i, :].reshape(patch_size, patch_size)
        hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
        feature_vec[i, :] = hog_features
    
    print time.time() - s
    outputs = clf.predict_proba(feature_vec.flatten())
    if outputs[0, 1] > outputs[0, 0]:
        num_correct += 1

print float(num_correct)/len(img_paths)
#print clf.predict_proba(feature_vec.flatten())
#print clf.predict(feature_vec.flatten())
#print clf.classes_

import pickle
import glob
import time

from sklearn import svm
from PIL import Image

from patch_extractor import *
from utils import *

f = open('classifier_svm_c1.pickle', 'r')
#f = open('classifier_adaboost.pickle', 'r')
clf = pickle.load(f)
f.close()

base_dir = 'cropped_images/newtest/'
#base_dir = 'negative_examples/'
img_paths = glob.glob(base_dir + '*.bmp')
img_paths = ['test.jpg']

num_correct = 0
for img_path in img_paths:
    img = Image.open(img_path).convert('L')
    num_bins = 10
    patch_size = 5
    stride = patch_size/2
    patch_extractor = PatchExtractor(patch_size, 1, stride=stride)

    angles, mags = get_mags_angles(img)
    angle_patches = patch_extractor.extract_all(angles)
    mag_patches = patch_extractor.extract_all(mags)
    feature_vec = np.zeros((angle_patches.shape[0], num_bins))
    s = time.time()
    for i in range(angle_patches.shape[0]):
        angle_patch = angle_patches[i, :].reshape(patch_size, patch_size)
        mag_patch = mag_patches[i, :].reshape(patch_size, patch_size)
        hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
        feature_vec[i, :] = hog_features
    print time.time() - s
    #outputs = clf.predict_proba(feature_vec.flatten())
    outputs = clf.predict(feature_vec.flatten())
    #print outputs
    #if outputs[0, 1] > outputs[0, 0]:
    if outputs == "True":
        num_correct += 1

print float(num_correct)/len(img_paths)
#print clf.predict_proba(feature_vec.flatten())
#print clf.predict(feature_vec.flatten())
#print clf.classes_

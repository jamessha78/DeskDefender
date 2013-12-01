import pickle
import numpy as np

from sklearn import svm
from PIL import Image

from patch_extractor import *
from utils import *

class Trainer(object):
    def __init__(self, directory):
        self.load_data(directory)
        self.clf = svm.SVC()

    def load_data(self, directory):
        test_image_1 = Image.open('images/newtest/ew-friends.gif').convert('L')
        test_image_1 = np.array(test_image_1)[:250, :250]
        test_image_2 = Image.open('images/newtest/ew-courtney-david.gif').convert('L')
        test_image_2 = np.array(test_image_2)[:250, :250]
        positive_images = [test_image_1]
        negative_images = [test_image_2]
        # Turn images from whatever directory structure into a list of images here
        num_bins = 10
        patch_size = 5
        patch_extractor = PatchExtractor(patch_size, 1)
        self.training_features = []
        self.training_labels = []
        # positive examples
        for image in positive_images:
            angles, mags = get_mags_angles(image)
            angle_patches = patch_extractor.extract_all(angles)
            mag_patches = patch_extractor.extract_all(mags)
            feature_vec = np.zeros((angle_patches.shape[0], num_bins))
            for i in range(angle_patches.shape[0]):
                angle_patch = angle_patches[i, :].reshape(patch_size, patch_size)
                mag_patch = mag_patches[i, :].reshape(patch_size, patch_size)
                hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
                feature_vec[i, :] = hog_features
            self.training_features.append(feature_vec.flatten().tolist())
            self.training_labels.append(1)
        # negative examples
        for image in negative_images:
            angles, mags = get_mags_angles(image)
            angle_patches = patch_extractor.extract_all(angles)
            mag_patches = patch_extractor.extract_all(mags)
            feature_vec = np.zeros((angle_patches.shape[0], num_bins))
            for i in range(angle_patches.shape[0]):
                angle_patch = angle_patches[i, :].reshape(patch_size, patch_size)
                mag_patch = mag_patches[i, :].reshape(patch_size, patch_size)
                hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
                feature_vec[i, :] = hog_features
            self.training_features.append(feature_vec.flatten().tolist())
            self.training_labels.append(0)
       
        self.training_features = np.array(self.training_features)
        self.training_labels = np.array(self.training_labels)

    def save_classifier(self):
        f = open('classifier.pickle', 'w+')
        pickle.dump(self.clf, f)
        f.close()

    def train(self):
        self.clf.fit(self.training_features, self.training_labels)

test = Trainer('')
test.train()
test.save_classifier()

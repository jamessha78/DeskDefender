import pickle
import glob
import numpy as np

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

from patch_extractor import *
from utils import *

class Trainer(object):
    def __init__(self, type='svm'):
        if type == 'svm':
            w = {'True': 50, 'False': 1}
            self.clf = svm.SVC(kernel='rbf', probability=True, class_weight=w)
        elif type == 'adaboost':
            self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                      algorithm="SAMME.R",
                                      n_estimators=200)
        else:
            print 'Unknown type', type
        self.type = type

    def load_data(self, pos_directory, neg_directory):
        print 'LOADING DATA...'
        positive_files = glob.glob(pos_dir + '*.bmp')
        negative_files = glob.glob(neg_dir + '*.bmp')
        #positive_files = positive_files[-1:]
        #negative_files = negative_files[-1:]
        #print positive_files
        #print negative_files
        positive_images = [Image.open(x).convert('L') for x in positive_files]
        negative_images = [Image.open(x).convert('L') for x in negative_files]
        # Turn images from whatever directory structure into a list of images here
        num_bins = 10
        patch_size = 20
        stride = patch_size/2
        patch_extractor = PatchExtractor(patch_size, 1, stride=stride)
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
            self.training_labels.append('True')
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
            self.training_labels.append('False')
       
        self.training_features = np.array(self.training_features)
        self.training_labels = np.array(self.training_labels)

    def save_classifier(self):
        print 'SAVING...'
        f = open('classifier_{0}.pickle'.format(self.type), 'w+')
        pickle.dump(self.clf, f)
        f.close()

    def train(self):
        print 'TRAINING...'
        self.clf.fit(self.training_features, self.training_labels)
        print 'Training error:', self.clf.score(self.training_features, self.training_labels)


pos_dir = 'cropped_images/test/'
neg_dir = 'negative_examples/'
test = Trainer('svm')
#test = Trainer('adaboost')
test.load_data(pos_dir, neg_dir)
test.train()
test.save_classifier()

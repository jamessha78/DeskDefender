import cProfile
from itertools import izip
import pickle
import glob

from sklearn import svm
import utils

try:
    from sklearn.ensemble import AdaBoostClassifier
except ImportError:
    pass  # Don't have new enough version of sklearn, will error if trying to make an adaboost classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from utils import *


TRUE = 1
FALSE = -1


class Cascade(object):
    def __init__(self, pos_directories, neg_directories, classifiers, thresholds, cell_size=6):
        self.pos_directories = pos_directories
        self.neg_directories = neg_directories
        self.window_shape = None
        self.cell_size = cell_size
        self.training_features = None
        self.training_labels = None
        self.classifiers = classifiers  # Classifiers in cascade, in order
        self.thresholds = thresholds  # Thresholds for the classifiers
        self.generate_cascade()

    def load_data(self):
        log("LOADING IMAGES...")
        positive_files = []
        negative_files = []
        for pos_dir in self.pos_directories:
            positive_files += glob.glob(pos_dir + '*')
        for neg_dir in self.neg_directories:
            negative_files = glob.glob(neg_dir + '*')
        positive_images = [np.array(Image.open(x).convert('L')) for x in positive_files]
        negative_images = [np.array(Image.open(x).convert('L')) for x in negative_files]

        image_shape = positive_images[0].shape
        for i, im in enumerate(positive_images):
            if im.shape != image_shape:
                message = "Not all images have same shape. Image with different shape: %s" % positive_files[i]
                raise ValueError(message)
        for i, im in enumerate(negative_images):
            if im.shape != image_shape:
                message = "Not all images have same shape. Image with different shape: %s" % negative_files[i]
                raise ValueError(message)
        self.window_shape = (image_shape[0] - 2, image_shape[1] - 2)  # Account for convolution

        log("EXTRACTING FEATURES...")
        self.training_features = np.vstack(
            (utils.extract_features(im, self.cell_size, self.window_shape)[0]
             for im in positive_images + negative_images)
        )
        self.training_labels = [TRUE] * len(positive_images) + [FALSE] * len(negative_images)
        self.training_labels = np.array(self.training_labels)
#         pickle.dump(self.training_features, open('cropped_training.pickle', 'w+'))
#         pickle.dump(self.training_labels, open('negative_training.pickle', 'w+'))
    def generate_cascade(self):
        if self.training_features is None:
            self.load_data()
        for i, classifier in enumerate(self.classifiers):
            log("Generating classifier for cascade step %s" % i)
            classifier.fit(self.training_features, self.training_labels)
            log("Training accuracy: %s" % classifier.score(self.training_features, self.training_labels))

    def save_cascade(self, name=None):
        log('SAVING...')
        self.training_features = None
        self.training_labels = None
        if name is None:
            name = "small_cascade.pickle"
        f = open(name, 'w+')
        pickle.dump(self, f)
        f.close()


def main():
    np.seterr(invalid='ignore')

    pos_dir = ['cropped_pubfig/']#, 'cropped_pubfig_eval/'] #['cropped_images/newtest/'] # , 'cropped_images/test-low/']
    neg_dir = ['negative_examples_pubfig/'] #, 'negative_examples/']

    estimators = [10]
    max_depths = [4]
    thresholds = [.3]
    win_sizes = [4]
    # estimators = [200]
    # max_depths = [10]
    # thresholds = [.5]

    classifiers = [
        RandomForestClassifier(criterion="entropy", n_jobs=-1, oob_score=True, n_estimators=est, max_depth=d)
        for est, d in izip(estimators, max_depths)
    ]

    cascade = Cascade(pos_dir, neg_dir, classifiers, thresholds)
    cascade.save_cascade()

    log("Out-of-bag error for each classifier: %s" % [c.oob_score_ for c in cascade.classifiers])
    log("DONE")


if __name__ == "__main__":
    #cProfile.run("main()", sort="tottime")
    main()

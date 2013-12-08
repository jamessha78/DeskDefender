import cProfile
import pickle
import glob

from sklearn import svm
try:
    from sklearn.ensemble import AdaBoostClassifier
except ImportError:
    pass  # Don't have new enough version of sklearn, will error if trying to make an adaboost classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from patch_extractor import *
from utils import *


TRUE = 1
FALSE = -1


class Cascade(object):
    def __init__(self, pos_directory, neg_directory, clf_type='svm'):
        self.clf_type = clf_type
        self.positive_images = []
        self.negative_images = []
        self.pos_directory = pos_directory
        self.neg_directory = neg_directory
        self.load_data()
        self.classifiers = {}  # Maps patch size to a classifier

    def load_data(self):
        log("LOADING IMAGES...")
        positive_files = glob.glob(self.pos_directory + '*.bmp')
        negative_files = glob.glob(self.neg_directory + '*.bmp')
        self.positive_images = [np.array(Image.open(x).convert('L')) for x in positive_files]
        self.negative_images = [np.array(Image.open(x).convert('L')) for x in negative_files]

    def generate_cascade(self, patch_sizes):
        if not self.positive_images or not self.negative_images:
            self.load_data()
        pool = multiprocessing.Pool(NUM_THREADS)
        for size in patch_sizes:
            log("Generating cascade for patch size %s" % size)
            self.add_classifier_for_patch_size(size, pool)
        pool.close()

    def add_classifier_for_patch_size(self, patch_size, pool=None):
        log("CREATING FEATURES...", False)
        # Turn images from whatever directory structure into a list of images here
        stride = patch_size/2
        patch_extractor = PatchExtractor(patch_size, 1, stride=stride)

        images = self.positive_images + self.negative_images
        training_features = extract_hog_features(images, patch_extractor, pool=pool)
        training_labels = [TRUE] * len(self.positive_images) + [FALSE] * len(self.negative_images)
        training_labels = np.array(training_labels)

        if training_features.shape[1] == 0:
            log("SKIPPING patch size = %s. No training features" % patch_size)
            return

        log("TRAINING...", False)
        classifier = self.get_new_classifier()
        classifier.fit(training_features, training_labels)
        #log("Training accuracy: %s" % classifier.score(training_features, training_labels))
        self.classifiers[patch_size] = classifier

    def get_new_classifier(self):
        if self.clf_type == 'svm':
            w = {TRUE: 50, FALSE: 1}
            return svm.SVC(kernel='rbf', probability=True, class_weight=w)
        elif self.clf_type == 'random_forest':
            return RandomForestClassifier(n_estimators=200, n_jobs=-1)
        elif type == 'adaboost':
            return AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                      algorithm="SAMME.R",
                                      n_estimators=200)
        else:
            raise ValueError("Unknown classifier type: %s" % self.clf_type)

    def get_classifiers_and_sizes(self):
        # Sort by patch size decreasing
        clfs_and_sizes = sorted(self.classifiers.iteritems(), key=lambda x: x[0], reverse=True)
        # Unzip to return a list of classifiers and a list of patch sizes
        sizes, classifiers = zip(*clfs_and_sizes)
        # Swap the order
        return classifiers, sizes

    def save_cascade(self, name=None):
        log('SAVING...')
        self.positive_images = []
        self.negative_images = []
        if name is None:
            name = 'cascade_{0}.pickle'.format(self.clf_type)
        f = open(name, 'w+')
        pickle.dump(self, f)
        f.close()


def main():
    np.seterr(invalid='ignore')

    pos_dir = 'cropped_images/test/'
    neg_dir = 'negative_examples/'

    #test = Cascade(pos_dir, neg_dir, 'svm')
    test = Cascade(pos_dir, neg_dir, 'random_forest')
    #test = Cascade(pos_dir, neg_dir, 'adaboost')

    #test.generate_cascade([50, 25, 20, 15, 10, 5, 2])
    #test.generate_cascade([50, 40, 30, 25, 20])
    #test.generate_cascade([20, 15, 10, 5])
    test.generate_cascade([20, 15, 10])

    test.save_cascade()
    log("DONE")


if __name__ == "__main__":
    #cProfile.run("main()", sort="tottime")
    main()
import cProfile
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

from patch_extractor import *
from utils import *


TRUE = 1
FALSE = -1


class Cascade(object):
    def __init__(self, pos_directory, neg_directory, clf_type='svm'):
        self.clf_type = clf_type
        self.pos_directory = pos_directory
        self.neg_directory = neg_directory
        self.training_features = None
        self.training_labels = None
        self.load_data()
        self.classifiers = {}  # Maps patch size to a classifier

    def load_data(self):
        log("LOADING IMAGES...")
        positive_files = glob.glob(self.pos_directory + '*.bmp')
        negative_files = glob.glob(self.neg_directory + '*.bmp')
        positive_images = [np.array(Image.open(x).convert('L')) for x in positive_files]
        negative_images = [np.array(Image.open(x).convert('L')) for x in negative_files]
        self.training_features = np.vstack(
            (utils.get_feature_matrix_from_image(im) for im in positive_images + negative_images)
        )
        self.training_labels = [TRUE] * len(positive_images) + [FALSE] * len(negative_images)
        self.training_labels = np.array(self.training_labels)
        log("Done loading images")

    def generate_cascade(self, patch_sizes):
        if self.training_features is None:
            self.load_data()
        pool = multiprocessing.Pool(NUM_THREADS)
        for size in patch_sizes:
            log("Generating cascade for patch size %s" % size)
            self.add_classifier_for_patch_size(size, pool)
        pool.close()

    def add_classifier_for_patch_size(self, patch_size):
        log("TRAINING...", False)
        classifier = self.get_new_classifier()
        classifier.fit(self.training_features, self.training_labels)
        print "Training accuracy", classifier.score(self.training_features, self.training_labels)
        correct_examples = self.training_labels == TRUE
        correct_features = self.training_features[correct_examples]
        correct_labels = self.training_labels[correct_examples]
        print "Training accuracy", classifier.score(self.training_features, self.training_labels)
        print "Training accuracy for positive examples", classifier.score(correct_features, correct_labels)
        #log("Training accuracy: %s" % classifier.score(training_features, training_labels))
        self.classifiers[patch_size] = classifier

    def get_new_classifier(self):
        if self.clf_type == 'svm':
            w = {TRUE: 50, FALSE: 1}
            return svm.SVC(kernel='rbf', probability=True, class_weight=w)
        elif self.clf_type == 'random_forest':
            return RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=10)
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
        self.training_features = None
        self.training_labels = None
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

    #test.generate_cascade([30, 25, 20, 15, 10, 5])
    #test.generate_cascade([20, 15, 10, 5])
    #test.generate_cascade([20, 15, 10])
    test.generate_cascade([5])

    test.save_cascade()
    log("DONE")


if __name__ == "__main__":
    #cProfile.run("main()", sort="tottime")
    main()
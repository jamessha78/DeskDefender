import multiprocessing
import pickle
import glob

from sklearn import svm
#from sklearn.ensemble import AdaBoostClassifier

from patch_extractor import *
from utils import *


NUM_THREADS = 8


def extract_features_from_image(image, patch_extractor, patch_size, num_bins):
    angles, mags = get_mags_angles(image)
    angle_patches = patch_extractor.extract_all(angles)
    mag_patches = patch_extractor.extract_all(mags)
    feature_vec = np.zeros((angle_patches.shape[0], num_bins))
    for i in range(angle_patches.shape[0]):
        angle_patch = angle_patches[i, :].reshape(patch_size, patch_size)
        mag_patch = mag_patches[i, :].reshape(patch_size, patch_size)
        hog_features = get_hog(angle_patch, mag_patch, bins=num_bins)
        feature_vec[i, :] = hog_features
    return feature_vec.flatten().tolist()


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
        print "LOADING IMAGES..."
        positive_files = glob.glob(self.pos_directory + '*.bmp')
        negative_files = glob.glob(self.neg_directory + '*.bmp')
        self.positive_images = [Image.open(x).convert('L') for x in positive_files]
        self.negative_images = [Image.open(x).convert('L') for x in negative_files]
        print "Done loading images"

    def generate_cascade(self, patch_sizes):
        if not self.positive_images or not self.negative_images:
            self.load_data()
        print "making pool"
        pool = multiprocessing.Pool(NUM_THREADS)
        for size in patch_sizes:
            print "Generating cascade for patch size", size
            self.add_classifier_for_patch_size(size, pool)
        pool.close()

    def add_classifier_for_patch_size(self, patch_size, pool=None):
        log("CREATING FEATURES...", False)
        # Turn images from whatever directory structure into a list of images here
        num_bins = 10
        stride = patch_size/2
        patch_extractor = PatchExtractor(patch_size, 1, stride=stride)
        #training_features = []
        #training_labels = []

        # Logic to make patch extraction multi-threaded
        make_own_pool = pool is None
        if make_own_pool:
            pool = multiprocessing.Pool(NUM_THREADS)
        data = self.positive_images + self.negative_images
        def data_iterator():
            for d in data:
                yield (d, patch_extractor, patch_size, num_bins)
        use_multithreading = False  # Not ready yet, issues with pickling images
        if use_multithreading:
            training_features = pool.map_async(extract_features_from_image, data_iterator()).get(99999)
        else:
            training_features = [extract_features_from_image(*d) for d in data_iterator()]
        training_labels = ['True'] * len(self.positive_images) + ['False'] * len(self.negative_images)

        ## positive examples
        #for image in self.positive_images:
        #    training_features.append(extract_features_from_image(image, patch_extractor, patch_size, num_bins))
        #    training_labels.append('True')
        ## negative examples
        #for image in self.negative_images:
        #    training_features.append(extract_features_from_image(image, patch_extractor, patch_size, num_bins))
        #    training_labels.append('False')
        if make_own_pool:
            pool.close()
       
        training_features = np.array(training_features)
        training_labels = np.array(training_labels)
        if training_features.shape[1] == 0:
            print "SKIPPING patch size = %s. No training features" % patch_size
            return

        log("TRAINING...", False)
        classifier = self.get_new_classifier()
        classifier.fit(training_features, training_labels)
        print 'Training accuracy:', classifier.score(training_features, training_labels)
        self.classifiers[patch_size] = classifier

    def get_new_classifier(self):
        if self.clf_type == 'svm':
            w = {'True': 50, 'False': 1}
            return svm.SVC(kernel='rbf', probability=True, class_weight=w)
        #elif type == 'adaboost':
        #    self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
        #                              algorithm="SAMME.R",
        #                              n_estimators=200)
        else:
            raise ValueError("Unknown classifier type: %s" % type)

    def get_classifiers_and_sizes(self):
        # Sort by patch size decreasing
        clfs_and_sizes = sorted(self.classifiers.iteritems(), key=lambda x: x[0], reverse=True)
        # Unzip to return a list of classifiers and a list of patch sizes
        sizes, classifiers = zip(*clfs_and_sizes)
        # Swap the order
        return classifiers, sizes

    def save_cascade(self, name=None):
        print 'SAVING...'
        self.positive_images = []
        self.negative_images = []
        if name is None:
            name = 'cascade_{0}.pickle'.format(self.clf_type)
        f = open(name, 'w+')
        pickle.dump(self, f)
        f.close()


def main():
    pos_dir = 'cropped_images/test/'
    neg_dir = 'negative_examples/'
    test = Cascade(pos_dir, neg_dir, 'svm')
    #test = Trainer('adaboost')
    #test.generate_cascade([50, 25, 20, 15, 10, 5, 2])
    test.generate_cascade([20, 15, 10])
    test.save_cascade()


if __name__ == "__main__":
    main()
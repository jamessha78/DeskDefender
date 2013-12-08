import cProfile
from joblib.parallel import multiprocessing
import time
import utils
import numpy as np
import matplotlib.pyplot
from PIL import Image
from PIL import ImageDraw

from patch_extractor import *
from sliding_window import *

class SVMClassifier:
    
    def __init__(self, clfs, rf_sizes):
        self.clfs = clfs
        self.rf_classes = rf_sizes
        self.patch_extractors = [None] * len(self.rf_classes)
        for i, rf_size in enumerate(rf_sizes):
            self.patch_extractors[i] = PatchExtractor(rf_size, 1, rf_size/2)
        self.sliding_window = SlidingWindow(1, [1, .5, .25], [5]*3)
        self.detections = []

    def test(self, im):
        pool = None
        if utils.USE_THREADING:
            pool = multiprocessing.Pool(utils.NUM_THREADS)

        start_time = time.time()
        utils.log_since("START TEST", start_time)
        patch_dicts = self.sliding_window.slide(im)
        for patch_extractor, classifier in zip(self.patch_extractors, self.clfs):
            utils.log_since("Extracting features for patch size %s" % patch_extractor.rf_size[0], start_time)
            patches = (pd['patch'] for pd in patch_dicts)
            features = utils.extract_hog_features(patches, patch_extractor, pool=pool)

            utils.log_since("Predicting for patch size %s" % patch_extractor.rf_size[0], start_time)
            outputs = classifier.predict(features)

            patch_dicts = [pd for i, pd in enumerate(patch_dicts) if outputs[i] == TRUE]
            if not patch_dicts:
                break  # Filtered out all potential patches

        self.detections = patch_dicts
        utils.log_since("DONE", start_time)
        if utils.USE_THREADING:
            pool.close()

    def draw(self, im):
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for detection in self.detections:
            pos = tuple(detection['position'])
            rf_size = detection['rf_size']
            draw.polygon(
                [
                    (pos[1], pos[0]),
                    (pos[1] + rf_size[1], pos[0]),
                    (pos[1] + rf_size[1], pos[0] + rf_size[0]),
                    (pos[1], pos[0] + rf_size[0])
                ],
                outline="#f00"
            )
        im.save('test.bmp')
        im.show()


from trainer import Cascade, TRUE  # Cascade needed for unpickling


def main():
    import pickle
    np.seterr(invalid='ignore')

    cascade = pickle.load(open('cascade_random_forest.pickle'))
    #cascade = pickle.load(open('cascade_svm.pickle'))

    classifiers, patch_sizes = cascade.get_classifiers_and_sizes()

    svm_classifier = SVMClassifier(classifiers[:-2], patch_sizes[:-2])
    #svm_classifier = SVMClassifier(classifiers, patch_sizes)

    im = Image.open('uncropped_images/newtest/ew-friends.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/harvard.gif').convert('L')

    im_np = np.array(im)
    svm_classifier.test(im_np)
    svm_classifier.draw(im)


if __name__ == '__main__':
    cProfile.run("main()", sort='tottime')
    #main()

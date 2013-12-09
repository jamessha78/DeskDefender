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
    
    def __init__(self, clfs, rf_sizes, image_size_hint=None):
        self.clfs = clfs
        self.rf_classes = rf_sizes
        self.patch_extractors = [None] * len(self.rf_classes)
        for i, rf_size in enumerate(rf_sizes):
            self.patch_extractors[i] = PatchExtractor(rf_size, 1, rf_size/2)

        if image_size_hint and False:  # Turn off for now, seems to make zoom slow
            num_window_sizes = 4
            target_width = 32
            target_height = 40
            image_size_y, image_size_x = image_size_hint

            min_scale = min(target_width / float(image_size_x), target_height / float(image_size_y))
            print "min_scale", min_scale
            scale_delta = (1 - min_scale) / float(num_window_sizes - 1)
            window_scales = [1 - i * scale_delta for i in range(num_window_sizes)]

            print window_scales
        else:
            window_scales = [1, .5, .25]
        self.scales = window_scales

        self.sliding_window = SlidingWindow(1, window_scales, [5]*len(window_scales))
        self.detections = []
        self.positions = None
        self.best_position = None

    def test(self, im):
        pool = None
        if utils.USE_THREADING:
            pool = multiprocessing.Pool(utils.NUM_THREADS)

        start_time = time.time()
        utils.log_since("Starting", start_time)
        all_features = []
        all_positions = []
        for scale in self.scales:
            if scale != 1:
                downsampled_image = ndimage.interpolation.zoom(im, scale)
            else:
                downsampled_image = im
            features, positions = utils.get_feature_matrix_from_image(downsampled_image)
            positions /= scale
            all_features.append(features)
            all_positions.append(positions)
        all_features = np.vstack(all_features)
        all_positions = np.vstack(all_positions)
        #print "features.shape", features.shape
        utils.log_since("Predicting", start_time)
        classifier = self.clfs[0]
        output_probs = classifier.predict_proba(all_features)[:, 1]
        print "non-zero", np.sum(output_probs != 0)
        hist, edges = np.histogram(output_probs, 100, (0, 1))
        for i, h in enumerate(hist):
            print "%.4f: %s" % (edges[i], "*" * h)
        meets_threshold = output_probs > .25
        output_probs = output_probs[meets_threshold]
        self.positions = all_positions[meets_threshold]

        if self.positions.shape[0]:
            best_activation = np.argmax(output_probs)
            self.best_position = self.positions[best_activation, :].tolist()
        else:
            self.best_position = None

        print "positions shape", self.positions.shape
        utils.log_since("Done", start_time)
        return self.best_position

        #patch_dicts = [pd for i, pd in enumerate(patch_dicts) if outputs[i] == TRUE]
        #if not patch_dicts:
        #    break  # Filtered out all potential patches



        utils.log_since("Sliding window", start_time)
        patch_dicts, images = self.sliding_window.slide(im)
        for patch_extractor, classifier in izip(self.patch_extractors, self.clfs):
            utils.log_since("Extracting features for patch size %s" % patch_extractor.rf_size[0], start_time)
            features = utils.extract_hog_features(patch_dicts, images, patch_extractor, pool=pool)

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
        if self.positions is not None:
            for top, left, bottom, right in self.positions:
                draw.polygon(
                    [
                        (left, top),
                        (right, top),
                        (right, bottom),
                        (left, bottom)
                    ],
                    outline="#0f0"
                )
        if self.best_position is not None:
            top, left, bottom, right = self.best_position
            draw.polygon(
                [
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom)
                ],
                outline="#f00"
            )
        #for detection in self.detections:
        #    pos = tuple(detection['orig_position'])
        #    rf_size = detection['orig_size']
        #    draw.polygon(
        #        [
        #            (pos[1], pos[0]),
        #            (pos[1] + rf_size[1], pos[0]),
        #            (pos[1] + rf_size[1], pos[0] + rf_size[0]),
        #            (pos[1], pos[0] + rf_size[0])
        #        ],
        #        outline="#f00"
        #    )
        im.save('test.bmp')
        im.show()


from trainer import Cascade, TRUE, FALSE  # Cascade needed for unpickling


def main():
    import pickle
    np.seterr(invalid='ignore')

    cascade = pickle.load(open('cascade_random_forest.pickle'))
    #cascade = pickle.load(open('cascade_svm.pickle'))
    classifiers, patch_sizes = cascade.get_classifiers_and_sizes()

    im = Image.open('test.png').convert('L')
    #im = Image.open('uncropped_images/newtest/bttf301.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/ew-friends.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/harvard.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/addams-family.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/audrey2.gif').convert('L')
    width, height = im.size

    start, end = 4, len(classifiers)
    #svm_classifier = SVMClassifier([classifiers[0], classifiers[3]], [patch_sizes[0], patch_sizes[3]])
    #svm_classifier = SVMClassifier(classifiers[start:end], patch_sizes[start:end], (height, width))
    svm_classifier = SVMClassifier(classifiers, patch_sizes, (height, width))

    im_np = np.array(im)
    #cProfile.runctx("svm_classifier.test(im_np)", None, locals(), sort='tottime')
    svm_classifier.test(im_np)
    svm_classifier.draw(im)


if __name__ == '__main__':
    #cProfile.run("main()", sort='tottime')
    main()

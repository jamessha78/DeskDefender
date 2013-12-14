import os
import cProfile
import multiprocessing
import time
import utils
import numpy as np
import matplotlib.pyplot
from PIL import Image
from PIL import ImageDraw

from patch_extractor import *
from sliding_window import *
import sys

class FaceDetector:
    
    def __init__(self, cascade, image_size_hint=None):
        self.classifiers = cascade.classifiers
        self.thresholds = cascade.thresholds
        self.cell_size = cascade.cell_size
        self.window_shape = cascade.window_shape

        if image_size_hint and False:
            num_window_sizes = 6
            target_width = 32
            target_height = 40
            image_size_y, image_size_x = image_size_hint

            min_scale = max(target_width / float(image_size_x), target_height / float(image_size_y))
            print "min_scale", min_scale
            scale_delta = (1 - min_scale) / float(num_window_sizes - 1)
            window_scales = [1 - i * scale_delta for i in range(num_window_sizes)]

        else:
            target_width = 32
            height, width = image_size_hint
            scales = [0.16, 0.33, 0.4, 0.5, 0.7, 0.8]
            offset = 4
            num_scales = 20
            scales = [float(x+offset)/(offset+num_scales) for x in xrange(num_scales)]
            window_scales = [target_width/(width * scale) for scale in scales]
#            window_scales = [2**(-x+2) for x in xrange(6)]
        self.scales = window_scales
        # self.scales = [1, .5, .25]

        self.positions = None
        self.likelihoods = None
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
            features, positions = utils.extract_features(downsampled_image, self.cell_size, self.window_shape)
            if features is None:
                continue
            positions /= scale
            all_features.append(features)
            all_positions.append(positions)

        all_features = np.vstack(all_features)
        all_positions = np.vstack(all_positions)
        for i, (classifier, threshold) in enumerate(izip(self.classifiers, self.thresholds)):
            utils.log_since("Testing cascade level %s" % i, start_time)
            output_probs = classifier.predict_proba(all_features)[:, 1]

            # print "non-zero", np.sum(output_probs != 0)
            # hist, edges = np.histogram(output_probs, 100, (0, 1))
            # for i, h in enumerate(hist):
            #     print "%.4f: %s" % (edges[i], "*" * h)

            meets_threshold = output_probs > threshold
            output_probs = output_probs[meets_threshold]
            all_positions = all_positions[meets_threshold]
            all_features = all_features[meets_threshold]

        self.positions = all_positions
        self.likelihoods = output_probs
        if self.positions.shape[0] > 0:
            best_activation = np.argmax(output_probs)
            self.best_position = self.positions[best_activation, :].tolist()
        else:
            self.best_position = None

        utils.log_since("Done", start_time)

        if utils.USE_THREADING:
            pool.close()

        return all_positions #self.best_position

    def draw(self, im):
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        if self.positions is not None and self.likelihoods is not None and len(self.likelihoods) > 1:
            pairs = izip(self.likelihoods, self.positions)
            sorted_pairs = sorted(pairs, key=lambda x: x[0])
            min_likelihood = np.min(self.likelihoods)
            max_likelihood = np.max(self.likelihoods)
            utils.log("Minimum displayed likelihood (blue): %s" % min_likelihood)
            utils.log("Maximum displayed likelihood (green): %s" % max_likelihood)
            for likelihood, (top, left, bottom, right) in sorted_pairs:
                # Color ranges from blue (not very likely match) to green (very likely match)
                likelihood_range = max_likelihood - min_likelihood
                normalized_likelihood = (likelihood - min_likelihood) / (1 if likelihood_range == 0 else likelihood_range)
                blue = hex(int((1 - normalized_likelihood) * 255))[2:]
                green = hex(int(normalized_likelihood * 255))[2:]
                if len(blue) == 1:
                    blue = "0" + blue
                if len(green) == 1:
                    green = "0" + green
                color = "#00%s%s" % (green, blue)
                draw.polygon(
                    [
                        (left, top),
                        (right, top),
                        (right, bottom),
                        (left, bottom)
                    ],
                    outline=color,
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

        im.save('test.bmp')
        im.show()


from trainer import Cascade # Cascade needed for unpickling


def main():
    import pickle
    np.seterr(invalid='ignore')

    cascade = pickle.load(open('small_cascade.pickle'))
    print len(cascade.classifiers)
    cascade.classifiers = [cascade.classifiers[-1]]
    print cascade.classifiers[0].oob_score_
    print cascade.classifiers[0]
    cascade.thresholds = [0.65, 0.45, .5]

    # im = Image.open('../getpubfig/dev/0be4da399ee374d95962a345c1f3c260.jpg').convert('L')
    dir_name = '../getpubfig/eval/'
#    dir_name = 'uncropped_images/newtest/'
#     im = Image.open('test.png').convert('L')
#     width, height = im.size
#     target_width = float(150)
#     im = ndimage.interpolation.zoom(im, target_width/width)
#     face_detector = FaceDetector(cascade, im.shape)
    
#     im_np = np.array(im)
#     #cProfile.runctx("svm_classifier.test(im_np)", None, locals(), sort='tottime')
#     face_detector.test(im_np)
#     face_detector.draw(Image.fromarray(im))
#     return
    for fname in os.listdir(dir_name):
        im = Image.open('%s%s'% (dir_name, fname)).convert('L')
    # im = Image.open('uncropped_images/newtest/police.gif').convert('L')
    # im = Image.open('uncropped_images/newtest/bttf301.gif').convert('L')
    # im = Image.open('uncropped_images/newtest/ew-friends.gif').convert('L')
    # im = Image.open('uncropped_images/newtest/harvard.gif').convert('L')
    # im = Image.open('uncropped_images/newtest/addams-family.gif').convert('L')
    # im = Image.open('uncropped_images/newtest/audrey2.gif').convert('L')
#        im = Image.open('test.png').convert('L')
        width, height = im.size
        target_width = float(70)
        im = ndimage.interpolation.zoom(im, target_width/width)
        face_detector = FaceDetector(cascade, im.shape)

        im_np = np.array(im)
    #cProfile.runctx("svm_classifier.test(im_np)", None, locals(), sort='tottime')
        face_detector.test(im_np)
        face_detector.draw(Image.fromarray(im))
    

if __name__ == '__main__':
    main()

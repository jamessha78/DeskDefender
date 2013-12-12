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

            print window_scales
        else:
            window_scales = [1, .5, .25]
        self.scales = window_scales
        # self.scales = [1, .5, .25]

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
            features, positions = utils.extract_features(downsampled_image, self.cell_size, self.window_shape)
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
        if self.positions.shape[0] > 0:
            best_activation = np.argmax(output_probs)
            self.best_position = self.positions[best_activation, :].tolist()
        else:
            self.best_position = None

        utils.log_since("Done", start_time)

        if utils.USE_THREADING:
            pool.close()

        return self.best_position

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

        im.save('test.bmp')
        im.show()


from trainer import Cascade # Cascade needed for unpickling


def main():
    import pickle
    np.seterr(invalid='ignore')

    cascade = pickle.load(open('cascade.pickle'))
    # cascade.classifiers = [cascade.classifiers[2]]
    cascade.thresholds = [.3]

    # im = Image.open('test.png').convert('L')
    # im = Image.open('uncropped_images/newtest/police.gif').convert('L')
    # im = Image.open('uncropped_images/newtest/bttf301.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/ew-friends.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/harvard.gif').convert('L')
    im = Image.open('uncropped_images/newtest/addams-family.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/audrey2.gif').convert('L')
    width, height = im.size

    face_detector = FaceDetector(cascade, (height, width))

    im_np = np.array(im)
    #cProfile.runctx("svm_classifier.test(im_np)", None, locals(), sort='tottime')
    face_detector.test(im_np)
    face_detector.draw(im)


if __name__ == '__main__':
    main()

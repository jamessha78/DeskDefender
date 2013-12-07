import cProfile
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
        self.sliding_window = SlidingWindow(1, [1, .5, .25], [10]*3)

    def test(self, im):
        patch_dicts = self.sliding_window.slide(im)
        print len(patch_dicts)
        new_patch_dicts = []
        for i, clf in enumerate(self.clfs):
            for patch_dict in patch_dicts:
                features = utils.extract_hog_features(
                    self.patch_extractors[i], patch_dict['patch']).flatten()
                output = clf.predict(features)
                if output == 'True':
                    new_patch_dicts.append(patch_dict)
                #new_patch_dicts.append(patch_dict)
                
            patch_dicts = new_patch_dicts
            new_patch_dicts = []

        self.detections = patch_dicts

    def draw(self, im):
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for detection in self.detections:
            pos = tuple(detection['position'])
            rf_size = detection['rf_size']
            draw.polygon([(pos[1], pos[0]), (pos[1]+rf_size[1], pos[0]), (pos[1]+rf_size[1], pos[0]+rf_size[0]), (pos[1], pos[0]+rf_size[0])], outline="#f00")
        im.save('test.bmp')
        im.show()


def main():
    from trainer import Cascade  # Needed for unpickling
    import pickle

    cascade = pickle.load(open('cascade_svm.pickle'))
    classifiers, patch_sizes = cascade.get_classifiers_and_sizes()
    svm_classifier = SVMClassifier(classifiers[:-1], patch_sizes[:-1])
    im = Image.open('uncropped_images/newtest/ew-friends.gif').convert('L')
    #im = Image.open('uncropped_images/newtest/harvard.gif').convert('L')
    im_np = np.array(im)
    svm_classifier.test(im_np)
    svm_classifier.draw(im)


if __name__ == '__main__':
    #cProfile.run("main()", sort='tottime')
    main()

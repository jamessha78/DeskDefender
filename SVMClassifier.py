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
        self.sliding_window = SlidingWindow(1, [1, .5, .25], [20]*3)

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
        draw = ImageDraw.Draw(im)
        for detection in self.detections:
            pos = tuple(detection['position'])
            rf_size = detection['rf_size']
            draw.polygon([(pos[1], pos[0]), (pos[1]+rf_size[1], pos[0]), (pos[1]+rf_size[1], pos[0]+rf_size[0]), (pos[1], pos[0]+rf_size[0])])
        to_show = np.array(im)
        matplotlib.pyplot.imsave('test.png', to_show)
         
if __name__ == '__main__':
    import pickle
    cls_0 = pickle.load(open('classifier_svm_c0.pickle'))
    cls_1 = pickle.load(open('classifier_svm_c1.pickle'))
    svm_classifier = SVMClassifier([cls_0, cls_1], [20,5])
    im = Image.open('uncropped_images/test/next.gif').convert('L')
    im_np = np.array(im)
    svm_classifier.test(im_np)
    svm_classifier.draw(im)
    

import utils
import numpy as np
import Image

class SVMClassifier:
    
    def __init__(self, clfs, rf_sizes):
        self.clfs = clfs
        self.rf_classes = rf_sizes
        self.patch_extractors = [None] * len(self.rf_classes)
        for i, rf_size in enumerate(rf_sizes):
            self.patch_extractors[i] = PatchExtractor(rf_size, 1, rf_size[0]/2)
        self.sliding_window = SlidingWindow(1, [1, .5, .25], [2]*3)

    def test(self, im):
        patch_dicts = self.sliding_window.slide(im)
        new_patch_dicts = []
        for i, clf in enumerate(self.clfs):
            for patch_dict in patch_dicts:
                features = utils.extract_hog_features(
                    self.patch_extractors[i], patch_dict['patch']).flatten()
                output = clf.predict(features)
                if outputs == 'True':
                    new_patch_dicts.append(patch_dict)
                
            patch_dicts = new_patch_dicts
            new_patch_dicts = []

        return patch_dicts

if __name__ == '__main__':
    import pickle
    cls_0 = pickle.load(open('classifier_svm_c0.pickle'))
    cls_1 = pickle.load(open('classifier_svm_c1.pickle'))
    svm_classifier = SVMClassifier([cls_0, cls_1], [10,5])
    print svm_classifier.test(np.array(Image.open('aerosmith-double.gif').convert('L')), normalize)
    
        
        

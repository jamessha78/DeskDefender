import numpy as np

class SlidingWindow(object):

    def __init__(self, num_channels, scales, strides, classifier, threshold=0, dtype=np.float32, training_im_size=[38, 30]):
        
        assert len(scales) == len(strides)
        
        self.scales = scales
        self.num_scales = len(scales)
        self.classifier = classifier
        self.threshold = threshold
        self.patch_extractors = []
        self.training_im_size = training_im_size
        for scale, stride in zip(scales, strides):
            rf_size = training_im_size
            patch_extractor = PatchExtractor(num_channels, rf_size, stride)
            self.patch_extractors.append(patch_extractor)


    def extract_all(self, image, normalize=False):
        patches = [None] * self.num_scales

        for i, patch_extractor in enumerate(self.patch_extractors):
            downsampled_image = ndimage.interpolation.zoom(image, self.scales[i])
            patches[i] = patch_extractor.extract_all(downsampled_image)
        return patches
        
    def patch_positions(self, image):
        positions = [None] * self.num_scales

        for i, patch_extractor in enumerate(self.patch_extractors):
            positions[i] = patch_extractor.patch_positions(image)
            
        return positions
        

    def slide(self, image):
        all_patches = self.extract_all(image)
        all_positions = self.patch_positions(image)
        
        patches_dicts = []
        for i, patches, positions in enumerate(zip(all_patches, all_positions)):
            rf_size = self.patch_extractor[i].rf_size
            scale = self.scales[i]
            for patch, position in zip(patches, positions):
                patches_dicts.append({ 
                        'patch' : patch,
                        'position' : position,
                        'rf_size' : [rf_size[0]/scale, rf_size[1]/scale]
                        })
        return patches_dicts
            
    
    if __name__ === '__main__':
        import pickle
        clf = pickle.load(

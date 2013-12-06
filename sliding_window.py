import numpy as np

import scipy.ndimage as ndimage
import matplotlib.pyplot

from patch_extractor import *

class SlidingWindow(object):

    def __init__(self, num_channels, scales, strides, threshold=0, dtype=np.float32, training_im_size=[40, 32]):
        
        assert len(scales) == len(strides)
        
        self.scales = scales
        self.num_scales = len(scales)
        self.threshold = threshold
        self.patch_extractors = []
        self.training_im_size = training_im_size
        for scale, stride in zip(scales, strides):
            rf_size = training_im_size
            patch_extractor = PatchExtractor(rf_size, num_channels, stride)
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
            downsampled_image = ndimage.interpolation.zoom(image, self.scales[i])
            positions[i] = patch_extractor.patch_positions(downsampled_image)
            
        return positions
        

    def slide(self, image):
        all_patches = self.extract_all(image)
        all_positions = self.patch_positions(image)
        
        patches_dicts = []
        for patches, positions, j in zip(all_patches, all_positions, range(len(all_patches))):
            rf_size = self.patch_extractors[j].rf_size
            scale = self.scales[j]
            for i in range(patches.shape[0]):
                patch = patches[i, :]
                position = positions[:, i]
                patches_dicts.append({ 
                        'patch' : patch.reshape(rf_size[0], rf_size[1]),
                        'position' : [position[0]/scale, position[1]/scale],
                        'rf_size' : [rf_size[0]/scale, rf_size[1]/scale]
                        })
            
        return patches_dicts
            

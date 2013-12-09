from itertools import izip
import numpy as np

import scipy.ndimage as ndimage
import matplotlib.pyplot

from patch_extractor import *
import utils


class SlidingWindow(object):

    def __init__(self, num_channels, scales, strides, threshold=0, dtype=np.float32, training_im_size=(38, 30)):
        
        assert len(scales) == len(strides)
        
        self.scales = scales
        self.num_scales = len(scales)
        self.threshold = threshold
        self.patch_extractors = []
        self.training_im_size = training_im_size
        for scale, stride in izip(scales, strides):
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
        """
        Extracts sliding windows over the preprocessed image
        @param image: A 2d ndarray representing an image
        @return: A length two tuple of a list of dictionaries, each corresponding to a window, and a list of images,
        each corresponding to a down sampled image. The down sampled images have been preprocessed by
        utils.preprocess_image
        """
        # Each 'patch' is a list of contributions to bins, like the output from utils.preprocess_image
        patches_dicts = []
        images = []
        for image_index, (scale, patch_extractor) in enumerate(izip(self.scales, self.patch_extractors)):
            if scale != 1:
                downsampled_image = ndimage.interpolation.zoom(image, scale)
            else:
                downsampled_image = image
            preprocessed_image = utils.preprocess_image(downsampled_image)
            images.append(preprocessed_image)
            positions = patch_extractor.patch_positions(preprocessed_image)
            rf_size = [patch_extractor.rf_size[0] / scale, patch_extractor.rf_size[1] / scale]
            #patches = patch_extractor.extract_all(preprocessed_image)
            #for i, patch in enumerate(patches):
            for i in xrange(positions.shape[1]):
                patch_position = positions[:, i].tolist()
                orig_position = (positions[:, i] / scale).tolist()
                patches_dicts.append({
                    'image_index': image_index,
                    'patch_position': patch_position,
                    'patch_size': patch_extractor.rf_size,
                    'orig_position': orig_position,
                    'orig_size': rf_size,
                })
        return patches_dicts, images

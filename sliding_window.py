import numpy as np

class SlidingWindow(object):

    def __init__(self, num_channels, rf_sizes, strides, classifier, threshold=0, dtype=np.float32):
        
        assert len(rf_sizes) == len(strides)
        
        self.num_scales = len(rf_sizes)
        self.classifier = classifier
        self.threshold = threshold
        self.patch_extractors = []
        for rf_size, stride in zip(rf_sizes, strides):
            patch_extractor = PatchExtractor(num_channels, rf_size, stride)
            self.patch_extractors.append(patch_extractor)


    def extract_all(self, image, normalize=False):
        patches = [None] * self.num_scales

        for i, patch_extractor in enumerate(self.patch_extractors):
            patches[i] = patch_extractor.extract_all(image)
            
        return patches
        
    def patch_positions(selfm image):
        positioins = [None] * self.num_scales

        for i, patch_extractor in enumerate(self.patch_extractors):
            positions[i] = patch_extractor.patch_positions(image)
            
        return positions
        


    def slide(self, image):
        all_patches = self.extract_all(image)
        all_positions = self.patch_positions(image)
        
        activations = []
        for patches, positions in zip(all_patches, all_positions):
            for patch, position in zip(patches, positions):
                activation = classifier(patch)
                if activation >= self.threshold:
                    activations.append((position, activation))

        return activations
            

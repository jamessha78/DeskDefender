import numpy as np
import Image

import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy import count_nonzero

class PatchExtractor(object):

  def __init__(self, rf_size, num_channels, stride=10, dtype=np.float32):

    if type(rf_size) is int:
      rf_size = [rf_size, rf_size]

    self.rf_size      = rf_size
    self.num_channels = num_channels
    self.stride       = stride
    self.dtype        = dtype

    self.dim = np.prod(self.rf_size) * num_channels

  def show_patches(self, patches, num_cols=15):
    patches = patches - patches.min()
    spacing = 2
    num_images = patches.shape[0]
    
    if num_cols > num_images:
      num_cols = num_images

    num_in_last_row = num_images % num_cols

    num_rows = num_images / num_cols

    if num_in_last_row != 0:
      num_rows += 1

    data_rows = self.rf_size[0] * num_rows + spacing * (num_rows - 1)
    data_cols = self.rf_size[1] * num_cols + spacing * (num_cols - 1)
    data = np.zeros((data_rows,data_cols))

    row_indices = range(0,data.shape[0], self.rf_size[0] + spacing)
    col_indices = range(0,data.shape[1], self.rf_size[1] + spacing)

    if self.num_channels == 1:
      size = tuple(self.rf_size)
    else:
      size = self.rf_size[:]
      size.append(self.num_channels)
      size = tuple(size)

    i = 0
    for row_idx in row_indices:
      for col_idx in col_indices:
        if i < num_images:
          data[row_idx:row_idx+self.rf_size[0], col_idx:col_idx+self.rf_size[1]] = patches[i].reshape(size)
          i += 1

    f1 = plt.imshow(data,interpolation='nearest')
    setp(f1.axes.get_xticklabels(), visible=False)
    setp(f1.axes.get_yticklabels(), visible=False)
    setp(f1.axes.xaxis.get_ticklines(), visible=False)
    setp(f1.axes.yaxis.get_ticklines(), visible=False)
    if self.num_channels == 1:
      f1.set_cmap('Greys_r')
    plt.show()

  def num_patches_for_image(self, image):
    return (image.shape[0] - self.rf_size[0] + 1) * (image.shape[1] - self.rf_size[1] + 1)

  def extract_all(self, image, normalize=False):

    rf_size = self.rf_size

    row_idxs = range(0, image.shape[0] - rf_size[0] + 1, self.stride)
    col_idxs = range(0, image.shape[1] - rf_size[1] + 1, self.stride)

    num_patches = len(row_idxs) * len(col_idxs)

    patches = np.empty((num_patches, self.dim),dtype=self.dtype)
    patch_num = 0
    for row in row_idxs:
      for col in col_idxs:
        patches[patch_num] = image[row:row+rf_size[0], col:col+rf_size[1]].flatten()
        patch_num += 1
    if normalize:
      self.normalize_patches(patches)

    return patches

  def patch_positions(self, image):

    rf_size = self.rf_size

    row_idxs = range(0, image.shape[0] - rf_size[0] + 1, self.stride)
    col_idxs = range(0, image.shape[1] - rf_size[1] + 1, self.stride)

    top     = np.tile(row_idxs, (len(col_idxs), 1)).T.flatten()
    bottom  = np.tile(col_idxs, len(row_idxs))

    top     = top     / np.float(image.shape[0] - rf_size[0] + 1)
    bottom  = bottom  / np.float(image.shape[1] - rf_size[1] + 1)

    positions = np.vstack((top, bottom))

    return positions

  # reg = regularization parameter
  def normalize_patches(self, patches, reg=10):
    patches -= np.mean(patches,axis=1).reshape( (patches.shape[0], 1) )
    patches /= (np.sqrt(np.mean(patches**2,axis=1))+reg).reshape((patches.shape[0],1))

    # Equivalent to the following, except for the regularization:
    # patches /= np.std(patches, acis=1).reshape( (patches.shape[0], 1) )

  def extract_all_file(self, fname, normalize=False):
    return self.extract_all(np.array(Image.open(fname).convert('L')), normalize)

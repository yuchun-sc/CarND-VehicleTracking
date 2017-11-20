import cv2
import glob
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)


color_space_range = {
    "RGB": [255, 255, 255],
    'HSV': [360, 1, 1],
    'HSL': [360, 1, 1]
}


class FeatureExtraction:
    def __init__(self):
        pass

    def run(self, image, size=(32, 32)):
        # read in png, which is already in range (0, 1)
        # read in jpg, will end up with range (0, 255)
        # image = image.astype(np.float32)/255
        feature_bin = self.get_bin_spatial(image, color_spaces=['YCrCb'], size=size)
        feature_hist = self.get_color_hist(image)
        feature_hog, _ = self.get_hog_features(image, size=size)
        feature = np.concatenate((feature_bin, feature_hist, feature_hog))

        return feature

    def get_bin_spatial(self, image, color_spaces=['RGB'], size=(32, 32)):
        # Convert image to new color space (if specified)
        feature = []
        for c_space in color_spaces:
            if c_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif c_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif c_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif c_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif c_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
            else:
                feature_image = np.copy(image)

            resized_image = cv2.resize(feature_image, size)

            bin_features = resized_image.ravel()

        return bin_features

    def get_color_hist(self, image, nbins=32, color_space='HSV'):
        # Compute the histogram of each channel 
        rge = color_space_range[color_space]

        rhist = np.histogram(image[:, :, 0], bins=nbins, range=(0, rge[0]))
        ghist = np.histogram(image[:, :, 1], bins=nbins, range=(0, rge[1]))
        bhist = np.histogram(image[:, :, 2], bins=nbins, range=(0, rge[2]))

        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

        return hist_features

    def get_hog_features(self, image, orient=9, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True, size=(32, 32)):
        """ generate HOG features
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        image = cv2.resize(image, size)

        hog_feature = []
        hog_image = []
        for channel in range(image.shape[2]):
            if vis:
                feature, hog_channel = hog(image[:,:,channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                           visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
                hog_image.append(hog_channel)
            else:
                feature = hog(image[:,:,channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2-Hys')
            hog_feature.append(feature)

        hog_feature = np.ravel(hog_feature)

        return hog_feature, hog_image


if __name__ == '__main__':
    pass




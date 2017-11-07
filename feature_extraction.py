import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import json
from multiprocessing import Pool


class FeatureExtraction:
    def __init__(self):
        pass

    def get_features(self, image, size=(32, 32)):

        image = image.astype(np.float32)/255

        feature_hist = self.get_bin_spatial_historgram(image, color_spaces=['YCrCb'], size=size)
        feature_hog = self.get_hog_features(image,size=size)
        feature = np.concatenate((feature_hist, feature_hog))
        return feature

    def get_bin_spatial_historgram(self, image, color_spaces=['RGB'], size=(32, 32)):
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
            # Use cv2.resize().ravel() to create the feature vector

            resized_image = cv2.resize(feature_image, size)
        # plt.imshow(resized_image)
        # plt.show()

            bin_features = resized_image.ravel()

            hist_features = self.color_hist(resized_image)
            feature = np.concatenate((feature, bin_features))
            feature = np.concatenate((feature, hist_features))
        # Return the feature vector0

        return self.normalize(feature)

    def color_hist(self, image, nbins=32):
        # image = mpimg.imread('cutout1.jpg')
        # Compute the histogram of the RGB channels separately

        rhist = np.histogram(image[:, :, 0], bins=nbins)
        ghist = np.histogram(image[:, :, 1], bins=nbins)
        bhist = np.histogram(image[:, :, 2], bins=nbins)
        # Generating bin centers
        # bin_edges = rhist[1]
        # bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, image, orient=9, pix_per_cell=8, cell_per_block=2, vis=True, feature_vec=True, size=(32, 32)):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        image = cv2.resize(image, size)
        hog_feature = []
        for i in range(3):
            hog_temp, hog_image = hog(image[:,:,i], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=vis, feature_vector=False, block_norm='L2-Hys')
            hog_feature.append(hog_temp.ravel())
        # hog2, hog_image = hog(image[:, :, 1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
        #                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
        #                       visualise=vis, feature_vector=False)
        # hog3, hog_image = hog(image[:, :, 2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
        #                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
        #                       visualise=vis, feature_vector=False)

        # feature = np.concatenate((hog1, hog2, hog3))
        feature = np.max(hog_feature, axis=0)
        # # print feature
        # plt.imshow(hog_image, cmap='gray')
        # plt.show()
        return self.normalize(feature)

    def normalize(self, X):
        X = X.reshape(-1, 1)
        X_scaler = StandardScaler().fit(X)
        result = X_scaler.transform(X).ravel()
        result = np.round(result, decimals=2)
        return result

if __name__ == '__main__':
    data_car = glob.glob("../Final-Project-data/vehicles/*/*.png")
    data_no_car = glob.glob("../Final-Project-data/non-vehicles/*/*.png")
    data = [data_car, data_no_car]

    obj = FeatureExtraction()

    def feature_ex(image_path):
        img = mpimg.imread(image_path)
        feature = obj.get_features(img)

        result = {}
        result['feature'] = list(feature)
        if 'non-vehicles/' in image_path:
            result['label'] = 0
        else:
            result['label'] = 1
        return result

    p = Pool(8)
    results = p.map(feature_ex, data_car)
    p.close()
    # print results
    with open('./features_YCrCb_32.json', 'w') as f:
        for line in results:
            f.write(json.dumps(line))
            f.write("\n")

    p = Pool(8)
    results = p.map(feature_ex, data_no_car)
    p.close()
    with open('./features_YCrCb_32.json', 'a') as f:
        for line in results:
            f.write(json.dumps(line))
            f.write("\n")


    # with open('./features_YCrCb_32.json', 'w') as f:
    #     count = 0
    #     for idx, folder in enumerate(data):
    #         for image_path in folder:
    #             try:
    #                 img = mpimg.imread(image_path)
    #                 feature = obj.get_features(img)
    #                 # print(feature.shape)
    #                 result = {}
    #                 result['feature'] = list(feature)
    #                 result['label'] = 1 - idx
    #                 f.write(json.dumps(result))
    #                 f.write("\n")
    #                 count += 1
    #             except:
    #                 print(image_path)
    #                 import traceback
    #                 traceback.print_exc()
    #             if count % 100 == 0:
    #                 print(count)



import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from feature_extraction import FeatureExtraction
import pickle


class ObjectionDetection:
    def __init__(self, model_file):
        pkl_file = open(model_file, 'rb')
        model_info = pickle.load(pkl_file)

        self.model = model_info['model']
        self.scaler = model_info['scaler']

    def run(self, img):
        window_params = self._get_window_params()
        windows = []
        for param in window_params:
            windows += self._generate_windowns(img, x_start_stop=param[0], y_start_stop=param[1],
                                     xy_window=param[2], xy_overlap=param[3])
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            # 4) Extract features for that window using single_img_features()
            features = FeatureExtraction().run(test_img)
            normalized_feature = self.scaler.transform(features)
            prediction = self.model.predict([normalized_feature])

            if prediction == 1:
                on_windows.append(window)

        resulting_img = self.draw_boxes(img, on_windows, color=(255, 0, 0), thick=2)

        return resulting_img, on_windows

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def _generate_windowns(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
        ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys * ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def _get_window_params(self):
        windows = []
        # append x_start_stop, y_start_stop, xy_window, xy_overlap
        windows.append([[400, 1000], [400, 464], (32, 32), (0.5, 0.5)])
        windows.append([[400, 1100], [400, 488], (48, 48), (0.5, 0.5)])
        windows.append([[400, 1200], [400, 496], (64, 64), (0.5, 0.5)])
        windows.append([[250, 1280], [400, 544], (96, 96), (0.75, 0.75)])
        windows.append([[250, 1280], [400, 544], (128, 128), (0.5, 0.5)])

        return windows

if __name__ == "__main__":
    # image_path = "./project_video/images00330.jpg"
    image_path = "/home/shaocheng/Desktop/test5.png"
    image = mpimg.imread(image_path)

    print(image)
    plt.imshow(image)
    plt.show()

    image = image.astype(np.float32) / 255

    pkl_file = open('./svm_model_linea_YCrCb_32.pkl', 'rb')

    clf = pickle.load(pkl_file)
    # a = FeatureExtraction().get_features(image)
    # print(clf.predict([a]))


    a = ObjectionDetection()
    param_list = []
    # append x_start_stop, y_start_stop, xy_window, xy_overlap
    param_list.append([[400, 1000], [400, 464], (32, 32), (0.5, 0.5)])
    param_list.append([[400, 1100], [400, 488], (48, 48), (0.5, 0.5)])
    param_list.append([[400, 1200], [400, 496], (64, 64), (0.5, 0.5)])
    param_list.append([[250, 1280], [400, 544], (96, 96), (0.75, 0.75)])
    param_list.append([[250, 1280], [400, 544], (128, 128), (0.5, 0.5)])
    # param_list.append([[250, 1280], [360, 544], (200, 112), (0.0, 0.75)])
    windows = []
    for param in param_list:
        windows += a.slide_window(image, x_start_stop=param[0], y_start_stop=param[1],
                                 xy_window=param[2], xy_overlap=param[3])

    window_img2 = a.draw_boxes(image, windows, color=(255, 0, 0), thick=2)
    plt.imshow(window_img2)
    plt.show()

    hot_windows = a.search_windows(image, windows, clf)

    window_img1 = a.draw_boxes(image, hot_windows, color=(255, 0, 0), thick=2)
    plt.imshow(window_img1)
    plt.show()


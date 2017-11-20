import numpy as np
import cv2
from scipy.ndimage.measurements import label


class PostProcessing:
    def __init__(self):
        pass

    def run(self, image, windows, threshold):
        heat_map, labels = self.generate_heat_window(image, windows, threshold)
        image, cars = self.draw_labeled_bboxes(image, labels)
        return image, cars

    def draw_labeled_bboxes(self, image, labels):
        # Iterate through all detected cars
        bboxes = []
        for car_index in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_index).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # reject bbox that has wierd ratio
            if (bbox[1][1] - bbox[0][1]) > (bbox[1][0] - bbox[0][0]) * 2 or (bbox[1][1] - bbox[0][1]) * 2 < (
                bbox[1][0] - bbox[0][0]):
                continue

            bboxes.append(bbox)
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (1, 0, 0), 4)
        # Return the image
        return image, bboxes

    def generate_heat_window(self, image, hot_windows, threshold):
        heat_image = self._add_heat(image, hot_windows)
        # Apply threshold to help remove false positives
        heat = self._apply_threshold(heat_image, threshold)
        # Visualize the heat map when displaying
        heat_map = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heat_map)

        return heat_map, labels

    def _add_heat(self, image, hot_windows):
        heat_image = np.zeros_like(image)
        # Iterate through list of bboxes
        for box in hot_windows:
            heat_image[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heat_map
        return heat_image

    def _apply_threshold(self, heat_image, threshold):
        # Zero out pixels below the threshold
        heat_image[heat_image <= threshold] = 0
        return heat_image

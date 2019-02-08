import itertools
import os
import sys

import cv2

from FoodConfig import FoodConfig
from FoodDataset import FoodDataset

ROOT_DIR = os.path.dirname(__file__) + os.sep + 'Mask_RCNN'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import model as modellib

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
from util import *


class FoodDetectionModel(object):
    def __init__(self, path=None):
        """ Get the most recent weight in the most recent log """
        if path is None:
            runs = [_ for _ in os.listdir(DEFAULT_LOGS_DIR) if _ != '.DS_Store']
            runs.sort(reverse=True)
            most_recent_log_dir = DEFAULT_LOGS_DIR + os.sep + runs[0] + os.sep
        else:
            most_recent_log_dir = DEFAULT_LOGS_DIR + os.sep + path + os.sep
        weights_path = most_recent_log_dir + [_ for _ in os.listdir(most_recent_log_dir)
                                              if _.__contains__('mask_rcnn_food_')][-1]

        """ Load the model configuration """

        class InferenceConfig(FoodConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.config = InferenceConfig()
        self.dataset = FoodDataset()
        self.dataset.load_food(skip_load=True)

        """ Load the model in inference mode """
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                       model_dir=DEFAULT_LOGS_DIR)

        print(f'Loading weights at {weights_path}')
        self.model.load_weights(weights_path, by_name=True)

    def predict(self, image: np.array, min_score_threshold=0.0, save_images=False, show_matplots=False) -> np.array:
        # Detect objects
        r = self.model.detect([image], verbose=1)[0]
        # Keep detections with scores higher than the min threshold
        keep_index = np.where(r['scores'] > min_score_threshold)[0]
        thres_r = {}
        for key in r:
            if key == 'masks':
                thres_r[key] = r[key][:, :, keep_index]
            else:
                thres_r[key] = r[key][keep_index]
        # Show the image, and the resulting image splash
        # if show_matplots:
        #     # Color splash
        #     splash = color_splash(image, thres_r['masks'])
        #     import matplotlib.pyplot as plt
        #     plt.imshow(image)
        #     plt.show()
        #     plt.imshow(splash)
        #     plt.title(f'Using scores {thres_r["scores"]}')
        #     plt.show()
        # Get the lines from the detected mask using connected components
        lines = self.get_lines(thres_r, image, show_matplots)
        return lines

    def get_lines(self, masks: np.array, image: np.array, show_matplots=False):
        """
         Using extreme point detection:
         https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        """
        image_copy = np.copy(image)
        lines = []
        points = []
        for i in range(masks['masks'].shape[2]):
            if masks['class_ids'][i] == 6:  # If the id is a 6, its a knife... so skip this
                continue
            # find contours in thresholded image, then grab the largest
            # one
            mask = np.array(masks['masks'][:, :, i], dtype=np.uint8)
            line_thickness = int(mask.shape[0] * 0.01)

            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1]
            c = max(cnts, key=cv2.contourArea)
            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            # draw the outline of the object, then draw each of the
            # extreme points, where the left-most is red, right-most
            # is green, top-most is blue, and bottom-most is teal
            cv2.drawContours(image_copy, [c], -1, (0, 255, 255), line_thickness)
            cv2.circle(image_copy, extLeft, line_thickness, (0, 0, 255), -1)
            cv2.circle(image_copy, extRight, line_thickness, (0, 255, 0), -1)
            cv2.circle(image_copy, extTop, line_thickness, (255, 0, 0), -1)
            cv2.circle(image_copy, extBot, line_thickness, (255, 255, 0), -1)

            """ Get the pairs of points with highest euclidean distance """
            combinations = list(itertools.combinations((extLeft, extRight, extBot, extTop), 2))
            # noinspection PyTypeChecker
            line = combinations[np.argmax([np.linalg.norm(np.subtract(line[0], line[1])) for line in combinations])]

            point_loc = (extLeft[0], extTop[1])
            points.append(point_loc)
            cv2.putText(img=image_copy, text=[_ for _ in self.dataset.class_info if _['id'] == masks['class_ids'][i]][0]['name'],
                        org=point_loc, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255), fontScale=int(line_thickness),
                        thickness=int(line_thickness))
            cv2.line(image_copy, line[0], line[1], (0, 0, 255), line_thickness)
            lines.append(line)

        # Show the image, and the resulting image splash
        if show_matplots:
            import matplotlib.pyplot as plt
            plt.imshow(image_copy)
            # plt.savefig('potato.jpg')
            foramatted_lines = "".join([str(_)+'\n' for _ in lines])
            plt.title(f'Predicted lines to cut are: {foramatted_lines}')
            plt.show()

        return lines, masks, points


if __name__ == '__main__':
    model = FoodDetectionModel('food20181210T0856')

    # Run some test images
    for image_path in [_ for _ in os.listdir(os.path.dirname(__file__) + os.sep + 'images') if _ != '.DS_Store']:
        image = skimage.io.imread(os.path.dirname(__file__) + os.sep + 'images' + os.sep + image_path)
        lines = model.predict(image, show_matplots=True)

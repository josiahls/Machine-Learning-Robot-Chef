import warnings

import skimage
from sklearn.model_selection import train_test_split

from Mask_RCNN.mrcnn import utils
import os
import numpy as np
import pandas as pd
import ast

############################################################
#  Dataset
############################################################
unique_id = 0

class FoodDataset(utils.Dataset):
    def load_food(self, dataset_dir='./data/EPIC_KITCHENS_2018/object_detection_images',
                  annotation_file='./annotations/EPIC_train_object_labels.csv', subset='test', skip_load=False):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("food", 1, "onion")
        self.add_class("food", 2, "potato")
        self.add_class("food", 3, "tomato")
        self.add_class("food", 4, "avocado")
        self.add_class("food", 5, "apple")
        self.add_class("food", 6, "knife")

        if skip_load:
            return

        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_subset_dir = os.path.join(os.path.dirname(__file__) + dataset_dir[1:], 'train')

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = None
        for i in range(2):
            try:
                annotations = pd.read_csv(os.path.dirname(__file__) + os.sep + subset + '.csv')
                break
            except FileNotFoundError:
                main_annotations = pd.read_csv(os.path.dirname(__file__) + annotation_file[1:])
                main_annotations = main_annotations[main_annotations['noun'].isin([class_name['name']
                                                                                   for class_name in self.class_info])]
                # And remove the ones that don't have any bounding boxes
                main_annotations = main_annotations[main_annotations['bounding_boxes'] != '[]']
                X_train, x_test = train_test_split(main_annotations, test_size=0.01)
                X_train.to_csv(os.path.dirname(__file__) + os.sep + 'train' + '.csv')
                x_test.to_csv(os.path.dirname(__file__) + os.sep + 'test' + '.csv')
                if i == 1:
                    raise FileNotFoundError

        # We convert the csv into a pandas data frame
        # annotations = pd.read_csv(os.path.dirname(__file__) + annotation_file[1:])
        # Keep the ones that have the classes we are looking for
        # annotations = annotations[annotations['noun'].isin([class_name['name'] for class_name in self.class_info])]
        # # And remove the ones that don't have any bounding boxes
        # annotations = annotations[annotations['bounding_boxes'] != '[]']
        # Finally Convert it to a dictionary
        annotations = annotations.to_dict('records')

        log = {'Images found: ': 0, 'Images not found: ': 0}
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = []
            class_ids = []
            if type(a['bounding_boxes']) is dict:
                for box in ast.literal_eval(a['bounding_boxes'].values()):
                    shape = {'all_points_x': [box[1], box[1] + box[3], box[1] + box[3], box[1]],
                             'all_points_y': [box[0], box[0], box[0] + box[2], box[0] + box[2]],
                             'name': 'polygon'}
                    polygons.append(shape)
                    class_ids.append([class_id['id'] for class_id in self.class_info if class_id['name'] == a['noun']][0])
            else:
                for box in ast.literal_eval(a['bounding_boxes']):
                    shape = {'all_points_x': [box[1], box[1] + box[3], box[1] + box[3], box[1]],
                             'all_points_y': [box[0], box[0], box[0] + box[2], box[0] + box[2]],
                             'name': 'polygon'}
                    polygons.append(shape)
                    class_ids.append([class_id['id'] for class_id in self.class_info if class_id['name'] == a['noun']][0])

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            filename = '0' * (10-len(str(a['frame']))) + str(a['frame']) + '.jpg'
            image_path = os.path.join(dataset_subset_dir, a['participant_id'], a['video_id'], filename)
            try:
                image = skimage.io.imread(image_path)
                log['Images found: '] += 1
            except FileNotFoundError:
                log['Images not found: '] += 1
                continue

            height, width = image.shape[:2]

            global unique_id
            self.add_image(
                'food',
                image_id=unique_id,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)
            unique_id += 1
            # break

        warnings.warn(f'The I found: {log["Images found: "]} images.')
        warnings.warn(f'The I did not find: {log["Images not found: "]} images.')

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] not in [class_name['source'] for class_name in self.class_info]:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.array(info['class_ids'])

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] in [class_name['name'] for class_name in self.class_info]:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

if __name__ == '__main__':
    d = FoodDataset()
    d.load_food('./data/EPIC_KITCHENS_2018/object_detection_images/',
                   './annotations/EPIC_train_object_labels.csv',
                   'train')
    d.load_food('./data/EPIC_KITCHENS_2018/object_detection_images/',
                   './annotations/EPIC_train_object_labels.csv',
                   'test')

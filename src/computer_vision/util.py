#!/usr/bin/env python
import os
from pathlib import Path
from typing import List
import warnings
import numpy as np
import skimage


def get_absolute_data_path(data_workspace: str = 'data', parent_up_limit=2, sub_folder_name: str = None):
    """
    Gets the absolute path for the 'data' directory.

    ***Note that this assumes the script is in the lowes-product-classifier***

    :param sub_folder_name: A sub folder name. It is the user's responsibility if the sub folder has separators such as
    '/' or '\'. Please use os.sep if this is the case.
    :param data_workspace: Name of the workspace. Default is 'data'
    :param parent_up_limit: The number of upper folders to look through to find the directory
    :return: The absolute path to the workspace. IE a string like:
    /Users/jlaivins/PycharmProjects/Lowes-Product-Classifier/lowes-product-classifier/data/
    """
    absolute_path = ''
    for i in range(-1, parent_up_limit):
        if i == -1:
            curr = str(Path().absolute())
        else:
            curr = str(Path().absolute().parents[i])
        if data_workspace in os.listdir(curr):
            absolute_path = curr + os.sep + data_workspace + os.sep
            break

    # If the user specifies a sub folder, add it
    if sub_folder_name is not None:
        absolute_path += sub_folder_name

    return absolute_path


def rename_filenames(files: List[str], data_workspace: str = 'data', prefix: str = 'im', postfix: str = '',
                     regex: str = '.'):
    # Get the absolute path to the data folder
    absolute_data_workspace_path = get_absolute_data_path(data_workspace)

    for file in files:
        # Get the file names in the directory
        document_names = os.listdir(absolute_data_workspace_path + file)
        increment = 0
        # Go through each image
        for document in document_names:
            full_path = absolute_data_workspace_path + file + os.sep + document
            # If the path exists
            if os.path.exists(full_path):
                rename(full_path, document, prefix, postfix, regex, increment)
            else:
                warnings.warn(f'File: {document} was not found in {full_path}. Verify that this file exists.',
                              RuntimeWarning)
            increment += 1


def rename(absolute_document_path: str, original_name: str, prefix: str, postfix: str, regex: str, inc: int = 0):
    # Get the new absolute path
    new_absolute_document_path = absolute_document_path[:-len(original_name)] + prefix + postfix + str(inc) + \
                                 original_name[original_name.index(regex):]

    print(f'Changing {absolute_document_path} to {new_absolute_document_path}')
    if not os.path.exists(new_absolute_document_path):
        os.rename(absolute_document_path, new_absolute_document_path)
    else:
        warnings.warn(f'The new file path: {new_absolute_document_path} already exists. Skipping renaming. ')

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

if __name__ == '__main__':
    # rename_filenames(['candle_holder', 'screw_sheet_metal_1in', 'screw_socket_1in'])
    get_absolute_data_path()
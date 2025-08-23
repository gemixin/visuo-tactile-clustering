"""
A collection of reusable functions for data loading and preprocessing.

Author: Gemma McLean
Date: August 2025
"""

import cv2
import os


def read_data_directory(data_directory, grayscale=False):
    """
    Reads images from the specified data directory and returns a list of images and their
    corresponding labels. Each subfolder in the data directory is treated as a label.

    Args:
        data_directory (str): Path to the data directory containing subfolders of images.
        grayscale (bool, optional): Whether to read images in grayscale. Defaults to False.

    Returns:
        images (list): List of images read from the data directory.
        labels (list): List of labels corresponding to each image, derived from
        the subfolder names.
    """

    # List to hold images and their corresponding labels
    images = []
    labels = []

    # Get all subfolders from provided folder
    subfolders = [f.path for f in os.scandir(data_directory) if f.is_dir()]
    # Iterate through each subfolder
    for subfolder in subfolders:
        # Get all images in the subfolder
        for filename in os.listdir(subfolder):
            # Check if the file is an image
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Read the image, optionally in grayscale
                if grayscale:
                    image = cv2.imread(os.path.join(subfolder, filename),
                                       cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(os.path.join(subfolder, filename))
                # Check if the image was read successfully
                if image is not None:
                    # Append the image to the list
                    images.append(image)
                    # Append the label (subfolder name) to the labels list
                    labels.append(os.path.basename(subfolder))

    # Return the list of images and their corresponding labels
    return images, labels

import cv2
import os


def read_data_folder(data_folder, grayscale=False):
    '''
    Reads images from the specified data folder and returns a list of images and their
    corresponding labels. Each subfolder in the data folder is treated as a label.
    Args:
        data_folder (str): Path to the data folder containing subfolders of images.
    Returns:
        images (list): List of images read from the data folder.
        labels (list): List of labels corresponding to each image, derived from
        the subfolder names.
    '''
    # List to hold images and their corresponding labels
    images = []
    labels = []

    # Get all subfolders from data folder
    data_folder = 'data'
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
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

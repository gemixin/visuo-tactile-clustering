import tsne_functions
import cv2
import os
import numpy as np

# List to hold images and their corresponding labels
labels = []
images = []

# Read in images from data folder and store them in a list
# Get all subfolders from data folder
data_folder = 'data'
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
# Iterate through each subfolder
for subfolder in subfolders:
    # Get all images in the subfolder
    for filename in os.listdir(subfolder):
        if filename.endswith('.jpg') or filename.endswith('.png'):

            image = cv2.imread(os.path.join(subfolder, filename), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Append the image to the list
                images.append(image)
                # Append the label (subfolder name) to the labels list
                labels.append(os.path.basename(subfolder))

# Use t-SNE to reduce the dimensionality of the images
images_flat = np.array([img.flatten() for img in images])

# Perform t-SNE
features_2d = tsne_functions.perform_tsne(images_flat)

# Plot the results
tsne_functions.plot_labels_2d(features_2d, labels)
tsne_functions.plot_objects_2d(features_2d, labels)

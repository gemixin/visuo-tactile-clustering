import cluster_functions
import data_functions
import numpy as np

'''
This script:
1. Reads tactile images from a data folder in grayscale
2. Flattens the images into 1D arrays for clustering
3. Uses PCA and/or t-SNE to reduce the dimensionality of the images
4. Visualises the results with labels and colors
'''

# Read images and labels from the data folder
print('Processing tactile images...')
images, labels = data_functions.read_data_folder('data', grayscale=True)
# Flatten the images for PCA/t-SNE
images_flat = np.array([img.flatten() for img in images])

print(images_flat.shape)  # Check the shape of the features
print(images_flat[:5])

# PCA
print('Performing PCA...')
features_2d = cluster_functions.pca_reduce(images_flat, 42)
# Plot the results
cluster_functions.plot_labels_2d('PCA', 'basic_plots', features_2d, labels, save=True)
cluster_functions.plot_objects_2d('PCA', 'basic_plots', features_2d, labels, save=True)

# t-SNE
print('Performing t-SNE...')
features_2d = cluster_functions.perform_tsne(images_flat, 42)
# Plot the results
cluster_functions.plot_labels_2d('t-SNE', 'basic_plots', features_2d, labels, save=True)
cluster_functions.plot_objects_2d('t-SNE', 'basic_plots', features_2d, labels, save=True)

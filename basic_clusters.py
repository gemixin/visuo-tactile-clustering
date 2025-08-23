"""
A script to cluster and visualise tactile images:
1. Reads tactile images from a data folder in grayscale
2. Flattens the images into 1D arrays for clustering
3. Uses PCA, t-SNE and UMAP to reduce the dimensionality of the images
4. Visualises the results with labels and colors

Author: Gemma McLean
Date: August 2025
"""

import cluster_functions
import data_functions
import numpy as np


# Read images and labels from the data folder
print('Processing tactile images...')
images, labels = data_functions.read_data_directory('data', grayscale=True)
# Flatten the images for PCA/t-SNE
images_flat = np.array([img.flatten() for img in images])

# Set random state for reproducibility
random_state = 42

# PCA
print('Performing PCA...')
features_2d = cluster_functions.pca_reduce(images_flat, random_state)
# Plot the results
cluster_functions.plot_labels_2d('PCA', 'plots/basic', features_2d, labels, save=True)
cluster_functions.plot_objects_2d('PCA', 'plots/basic', features_2d, labels, save=True)

# t-SNE
print('Performing t-SNE...')
features_2d = cluster_functions.perform_tsne(images_flat, random_state)
# Plot the results
cluster_functions.plot_labels_2d('t-SNE', 'plots/basic', features_2d, labels, save=True)
cluster_functions.plot_objects_2d('t-SNE', 'plots/basic', features_2d, labels, save=True)

# UMAP
print('Performing UMAP...')
features_2d = cluster_functions.perform_umap(images_flat, random_state)
# Plot the results
cluster_functions.plot_labels_2d('UMAP', 'plots/basic', features_2d, labels, save=True)
cluster_functions.plot_objects_2d('UMAP', 'plots/basic', features_2d, labels, save=True)

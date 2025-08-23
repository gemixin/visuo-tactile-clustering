"""
A script to cluster and visualise tactile images:
1. Reads tactile images from a data folder in grayscale
2. Extracts features from the images using a pre-trained model
3. Uses PCA, t-SNE and UMAP to reduce the dimensionality of the images
4. Visualises the results with labels and colors

Author: Gemma McLean
Date: August 2025
"""

from feature_extractor import FeatureExtractor
import cluster_functions


# Initialise feature extractor
feature_extractor = FeatureExtractor()

# Extract features from all images
print("Extracting features from tactile images...")
features, labels, image_paths = feature_extractor.extract_features_from_directory(
    'data')

print(f"Extracted features from {len(features)} images")
print(f"Feature vector size: {features.shape[1]}")
print(f"Object types found: {set(labels)}")

# PCA
print('Performing PCA...')
features_2d = cluster_functions.pca_reduce(features, 42)
# Plot the results
cluster_functions.plot_labels_2d('PCA', 'plots/ml', features_2d, labels)
cluster_functions.plot_objects_2d('PCA', 'plots/ml', features_2d, labels)

# t-SNE
print('Performing t-SNE...')
features_2d = cluster_functions.perform_tsne(features, 42)
# Plot the results
cluster_functions.plot_labels_2d('t-SNE', 'plots/ml', features_2d, labels)
cluster_functions.plot_objects_2d('t-SNE', 'plots/ml', features_2d, labels)

# UMAP
print('Performing UMAP...')
features_2d = cluster_functions.perform_umap(features, 42)
# Plot the results
cluster_functions.plot_labels_2d('UMAP', 'plots/ml', features_2d, labels)
cluster_functions.plot_objects_2d('UMAP', 'plots/ml', features_2d, labels)

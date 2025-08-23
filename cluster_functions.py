"""
A collection of reusable functions for clustering and visualising the data.

Author: Gemma McLean
Date: August 2025
"""

import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import colorsys
from sklearn.manifold import TSNE
import umap

# Define base colors for main categories
BASE_COLORS = {
    'banana': '#E5FF00',
    'beans': '#0037AD',
    'ball': '#009C34',
    'pringles': '#FF0000',
    'hammer': '#BF20FD'
}

# Fixed colors for object sublabels
FIXED_COLORS = ['#e41a1c', '#4daf4a', '#377eb8']


def get_lighter_color(hex_color):
    """
    Generate a lighter shade of a given hex color.

    Args:
        hex_color (str): Base color in hex format.

    Returns:
        str: Lighter shade of the base color in hex format.
    """

    # Convert to HLS
    rgb = to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Increase lightness (ensuring it does not exceed 1.0)
    new_l = min(l + 0.2, 1.0)
    # Convert back to RGB
    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return to_hex(new_rgb)


def get_darker_color(hex_color):
    """
    Generate a darker shade of a given hex color.

    Args:
        hex_color (str): Base color in hex format.

    Returns:
        str: Darker shade of the base color in hex format.
    """

    # Convert to HLS
    rgb = to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Decrease lightness (ensuring it does not go below 0.0)
    new_l = max(l - 0.2, 0.0)
    # Convert back to RGB
    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return to_hex(new_rgb)


def get_color_map(labels):
    """
    Create a color map for labels based on their main categories.

    Args:
        labels (list): List of labels.

    Returns:
        dict: Dictionary mapping labels to colors.
    """

    # Create an empty color map
    color_map = {}
    # For each label
    for i, sub in enumerate(labels):
        main_label = sub.split('_')[0]
        if main_label in BASE_COLORS:
            # Assign the main colour to the first
            if i % 3 == 0:
                color_map[sub] = BASE_COLORS[main_label]
            # Assign the lighter colour to the second
            elif i % 3 == 1:
                color_map[sub] = get_lighter_color(BASE_COLORS[main_label])
            # Assign the darker colour to the third
            else:
                color_map[sub] = get_darker_color(BASE_COLORS[main_label])
    return color_map


def pca_reduce(features, random_state):
    """
    Perform PCA on the given features to reduce to 2 components.

    Args:
        features (array-like): List or array of features to reduce.
        random_state (int): Random state for reproducibility.

    Returns:
        ndarray: 2D array of PCA reduced features.
    """

    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(features)


def perform_tsne(features, random_state):
    """
    Perform t-SNE on the given features to reduce to 2 components.

    Args:
        features (array-like): List or array of features to reduce.
        random_state (int): Random state for reproducibility.

    Returns:
        ndarray: 2D array of t-SNE transformed features.
    """

    tsne = TSNE(n_components=2, random_state=random_state)
    return tsne.fit_transform(features)


def perform_umap(features, random_state):
    """
    Perform UMAP on the given features to reduce to 2 components.

    Args:
        features (array-like): List or array of features to reduce.
        random_state (int): Random state for reproducibility.

    Returns:
        ndarray: 2D array of UMAP transformed features.
    """

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    return reducer.fit_transform(features)


def plot_labels_2d(type, save_dir, features_2d, labels, silhouette=True):
    """
    Plot all labels of all objects in a 2D space. Each object has its own main colour.
    Each of the 3 sublabels for each object has a different shade of the main color.
    Prints silhouette score for all labels.

    Args:
        type (str): 'PCA', 't-SNE' or 'UMAP'.
        save_dir (str): Directory to save the plots.
        features_2d (ndarray): 2D array of features to plot.
        labels (list): List of labels corresponding to each feature.
        silhouette (bool): Whether to compute silhouette score.
        save (bool): Whether to save the plot.
    """

    # Get unique labels and create a color map
    sorted_labels = sorted(set(labels))
    color_map = get_color_map(sorted_labels)

    # Plot all points with their corresponding colors
    plt.figure(figsize=(10, 10))
    # Iterate through each label and plot the points
    for lab in sorted_labels:
        # Get indices of points for this label
        indices = [i for i, l in enumerate(labels) if l == lab]
        # Plot the points for this label
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                    label=lab, color=color_map[lab], alpha=0.7)

    # Set title and labels for the plot
    plt.title(f'{type} visualisation of all labels')
    plt.xlabel(f'{type} component 1')
    plt.ylabel(f'{type} component 2')
    plt.legend()

    # Encode labels as numbers for silhouette score calculation
    le = LabelEncoder()
    cluster_labels = le.fit_transform(labels)
    # Calculate silhouette score
    if silhouette and len(set(cluster_labels)) > 1 and len(cluster_labels) > 1:
        score = silhouette_score(features_2d, cluster_labels)
        print(f'{type} Silhouette score for all labels: {score:.3f}')

    # If the save directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Save plot
    plt.savefig(f'{save_dir}/{type}_labels_plot.png')
    print(f'Plot saved as {save_dir}/{type}_labels_plot.png.')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_objects_2d(type, save_dir, features_2d, labels, silhouette=True):
    """
    Plot each object and its 3 sublabels in its own 2D space.
    Each object uses a fixed red, green, blue color for labelling.
    Prints the silhouette score for each main class.

    Args:
        type (str): 'PCA', 't-SNE' or 'UMAP'.
        save_dir (str): Directory to save the plots.
        features_2d (ndarray): 2D array of features to plot.
        labels (list): List of labels corresponding to each feature.
        silhouette (bool): Whether to compute silhouette scores.
    """

    # Get unique labels and main labels
    sorted_labels = sorted(set(labels))
    main_labels = list(BASE_COLORS.keys())

    # Create the subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot each object with its own sublabels
    for idx, main in enumerate(main_labels):
        # Current axis
        ax = axes[idx]
        # Get sublabels for the current main label
        sublist = [s for s in sorted_labels if s.startswith(main)]
        all_indices = []
        cluster_labels = []
        # For each sublabel, plot the points
        for i, sub in enumerate(sublist):
            # Cycle through fixed colors red, green, blue
            color = FIXED_COLORS[i % 3]
            # Get indices of points for this sublabel
            indices = [j for j, l in enumerate(labels) if l == sub]
            ax.scatter(features_2d[indices, 0], features_2d[indices, 1],
                       label=sub, color=color, alpha=0.7)
            # Collect indices and labels for silhouette score calculation
            all_indices.extend(indices)
            cluster_labels.extend([i] * len(indices))

        # Set title and labels for the axis
        ax.set_title(f'{main.capitalize()} sublabels')
        ax.set_xlabel(f'{type} component 1')
        ax.set_ylabel(f'{type} component 2')
        ax.legend()

        # Calculate silhouette score for this main class
        if silhouette and len(set(cluster_labels)) > 1 and len(all_indices) > 1:
            score = silhouette_score(features_2d[all_indices], cluster_labels)
            print(f'{type} Silhouette score for {main}: {score:.3f}')

    # Hide the 6th subplot as unused
    fig.delaxes(axes[5])

    # If the save directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Save plot
    plt.savefig(f'{save_dir}/{type}_objects_plot.png')
    print(f'Plot saved as {save_dir}/{type}_objects_plot.png.')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

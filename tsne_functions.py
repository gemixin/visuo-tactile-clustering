import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import colorsys
from sklearn.manifold import TSNE

# Define base colors for main categories
BASE_COLORS = {
    'banana': "#E5FF00",
    'beans': "#0037AD",
    'ball': "#009C34",
    'pringles': "#FF0000",
    'hammer': "#BF20FD"
}

# Fixed colors for object sublabels
FIXED_COLORS = ["#e41a1c", "#4daf4a", "#377eb8"]


def get_lighter_color(hex_color):
    """
    Generate a lighter shade of a given hex color.
    :param hex_color: Base color in hex format.
    :return: Lighter shade of the base color in hex format.
    """
    rgb = to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    new_l = min(l + 0.2, 1.0)  # Ensure lightness does not exceed 1.0
    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return to_hex(new_rgb)


def get_darker_color(hex_color):
    """
    Generate a darker shade of a given hex color.
    :param hex_color: Base color in hex format.
    :return: Darker shade of the base color in hex format.
    """
    rgb = to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    new_l = max(l - 0.2, 0.0)  # Ensure lightness does not go below 0.0
    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return to_hex(new_rgb)


def get_color_map(labels):
    """
    Create a color map for labels based on their main categories.
    :param labels: List of labels.
    :return: Dictionary mapping labels to colors.
    """
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


def perform_tsne(features):
    """
    Perform t-SNE on the given features.
    :param features: List or array of features to reduce.
    :return: 2D array of t-SNE transformed features.
    """
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(features)


def plot_labels_2d(features_2d, labels):
    """
    Plot all labels of all objects in a 2D space.
    Each object has it's own main colour.
    Each of the 3 sublabels for each object has a different shade of the main color.
    :param features_2d: 2D array of features to plot.
    :param labels: List of labels corresponding to each feature.
    """
    # Get unique labels and create a color map
    sorted_labels = sorted(set(labels))
    color_map = get_color_map(sorted_labels)

    # Plot all points with their corresponding colors
    plt.figure(figsize=(10, 10))
    for lab in sorted_labels:
        indices = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                    label=lab, color=color_map[lab], alpha=0.7)
    plt.title("t-SNE visualisation of all labels")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_objects_2d(features_2d, labels):
    """
    Plot each object and it's 3 sublabels in its own 2D space.
    Each object uses a fixed red, green, blue color for labelling.
    :param features_2d: 2D array of features to plot.
    :param labels: List of labels corresponding to each feature.
    """
    # Get unique labels and main labels
    sorted_labels = sorted(set(labels))
    main_labels = list(BASE_COLORS.keys())

    # Create the subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot each object with its own sublabels
    for idx, main in enumerate(main_labels):
        ax = axes[idx]
        sublist = [s for s in sorted_labels if s.startswith(main)]
        for i, sub in enumerate(sublist):
            # Cycle through fixed colors for sublabels
            color = FIXED_COLORS[i % 3]
            indices = [j for j, l in enumerate(labels) if l == sub]
            ax.scatter(features_2d[indices, 0], features_2d[indices, 1],
                       label=sub, color=color, alpha=0.7)
        ax.set_title(f"{main.capitalize()} sublabels")
        ax.set_xlabel('t-SNE component 1')
        ax.set_ylabel('t-SNE component 2')
        ax.legend()

    # Hide the 6th subplot as unused
    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.show()

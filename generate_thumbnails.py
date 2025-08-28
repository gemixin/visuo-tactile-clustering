"""
A script to save a plot of thumbnails for each object category, randomly selecting 5 images
for each sublabel.

Author: Gemma McLean
Date: August 2025
"""

import random
import matplotlib.pyplot as plt
import data_functions


# Read images and labels from the data folder
print('Processing tactile images...')
images, labels = data_functions.read_data_directory('data')

# Define object categories and their sublabels
objects = [
    ('ball', ['ball_logo', 'ball_seam', 'ball_surface']),
    ('banana', ['banana_skin', 'banana_stem', 'banana_tip']),
    ('beans', ['beans_base', 'beans_body', 'beans_lid']),
    ('hammer', ['hammer_handle', 'hammer_head', 'hammer_neck']),
    ('pringles', ['pringles_base', 'pringles_body', 'pringles_lid'])
]

# Create the thumbnails
print('Creating thumbnails...')
# Loop through each object category and its sublabels
for main, sublabels in objects:
    print(f'Creating thumbnails for {main}...')
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    # Move the suptitle closer to the content
    fig.suptitle(main.capitalize(), fontsize=18)
    # For each sublabel, create a row of thumbnails
    for row, sub in enumerate(sublabels):
        # Get all images for this sublabel
        sub_imgs = [img for img, lab in zip(images, labels) if lab == sub]
        # Randomly select 5 images (or all if fewer than 5)
        chosen = random.sample(sub_imgs, min(5, len(sub_imgs)))
        # For each chosen image, create a thumbnail
        for col, img in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(sub, fontsize=10)
            ax.axis('off')
    # Save the plot
    plt.savefig(f'thumbnails/{main}_thumbnails.png', dpi=300, bbox_inches='tight')
    print(f'Plot saved as thumbnails/{main}_thumbnails.png.')

print('Finished!')

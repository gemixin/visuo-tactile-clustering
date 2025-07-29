import random
import matplotlib.pyplot as plt
import data_functions

'''
Save a plot of thumbnails for each object category.
Randomly select 5 images for each sublabel.
'''

# Read images and labels from the data folder
print('Processing tactile images...')
images, labels = data_functions.read_data_folder('data')

objects = [
    ('ball', ['ball_logo', 'ball_seam', 'ball_surface']),
    ('banana', ['banana_skin', 'banana_stem', 'banana_tip']),
    ('beans', ['beans_base', 'beans_body', 'beans_lid']),
    ('hammer', ['hammer_handle', 'hammer_head', 'hammer_neck']),
    ('pringles', ['pringles_base', 'pringles_body', 'pringles_lid'])
]

print('Creating plots...')
for main, sublabels in objects:
    print(f'Creating thumbnails for {main}...')
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    # Move the suptitle closer to the content
    fig.suptitle(main.capitalize(), fontsize=18)
    for row, sub in enumerate(sublabels):
        # Get all images for this sublabel
        sub_imgs = [img for img, lab in zip(images, labels) if lab == sub]
        # Randomly select 5 images (or all if fewer than 5)
        chosen = random.sample(sub_imgs, min(5, len(sub_imgs)))
        for col, img in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(sub, fontsize=10)
            ax.axis('off')
    # Save the plot
    plt.savefig(f'thumbnails/{main}_thumbnails.png')
    print(f'Plot saved as thumbnails/{main}_thumbnails.png.')

print('Finished!')

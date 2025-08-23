from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import numpy as np


class FeatureExtractor:
    """
    A class for extracting features from images using a pre-trained ResNet model.

    Author: Gemma McLean
    Date: August 2025
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialise the feature extractor.

        Args:
            device (str, optional): Device to run the model on. Defaults to 'cuda' if
            available, else 'cpu'.
        """

        # Set device
        self.device = device

        # Load pre-trained ResNet50 model
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the final classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            # Padding to maintain aspect ratio
            transforms.Pad((0, 40, 0, 40), fill=0, padding_mode='constant'),
            # Resize to 224x224
            transforms.Resize((224, 224)),
            # Convert to tensor
            transforms.ToTensor(),
            # ImageNet normalisation
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f'Feature extractor initialized on device: {self.device}')

    def extract_features_from_image(self, image_path):
        """
        Extract features from a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            features (np.ndarray): Extracted features from the image.
        """

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract and return features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            return features
        except Exception as e:
            # Handle exceptions
            print(f'Error processing {image_path}: {e}')
            return None

    def extract_features_from_directory(self, data_dir):
        """
        Extract features from all images in the data directory.

        Args:
            data_dir (str): Path to the data directory containing subfolders of images.

        Returns:
            features (np.ndarray): Array of extracted features.
            labels (list): List of labels corresponding to each image.
            image_paths (list): List of image paths.
        """

        # Get path of data directory
        data_dir = Path(data_dir)

        # Create lists to store features, labels, and image paths
        features_list = []
        labels_list = []
        image_paths = []

        # Process each object directory
        for object_dir in data_dir.iterdir():
            # Check if the path is a directory
            if object_dir.is_dir():
                # Get the object name
                object_name = object_dir.name
                print(f'Processing {object_name}...')

                # Find all image files
                image_files = list(object_dir.glob('*.jpg')) + \
                    list(object_dir.glob('*.png'))

                # For each image file
                for img_path in image_files:
                    # Extract features
                    features = self.extract_features_from_image(img_path)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(object_name)
                        image_paths.append(str(img_path))

        # Return the extracted features, labels, and image paths
        return np.array(features_list), labels_list, image_paths

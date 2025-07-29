"""
Tactile Image Clustering with t-SNE Visualization

This script extracts features from tactile images using a pre-trained ResNet model
and visualizes the clusters using t-SNE, grouped by object type.
"""

from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import cluster_functions
import numpy as np


class FeatureExtractor:
    """Extract features from images using a pre-trained ResNet model."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Load pre-trained ResNet50 model
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the final classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Pad((0, 40, 0, 40), fill=0, padding_mode='constant'),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        print(f"Feature extractor initialized on device: {self.device}")

    def extract_features_from_image(self, image_path):
        """Extract features from a single image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()

            return features
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def extract_features_from_directory(self, data_dir):
        """Extract features from all images in the data directory."""
        data_dir = Path(data_dir)
        features_list = []
        labels_list = []
        image_paths = []

        # Process each object directory
        for object_dir in data_dir.iterdir():
            if object_dir.is_dir():
                object_name = object_dir.name
                print(f"Processing {object_name}...")

                # Find all image files
                image_files = list(object_dir.glob("*.jpg")) + \
                    list(object_dir.glob("*.png"))

                for img_path in image_files:
                    features = self.extract_features_from_image(img_path)
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(object_name)
                        image_paths.append(str(img_path))

        return np.array(features_list), labels_list, image_paths


def main():
    """Main function to run the complete analysis."""
    # Set up paths
    data_dir = Path("data")

    if not data_dir.exists():
        print(f"Data directory '{data_dir}' not found!")
        return

    print("Starting Tactile Image Clustering Analysis")
    print("="*50)

    # Initialize feature extractor
    feature_extractor = FeatureExtractor()

    # Extract features from all images
    print("Extracting features from images...")
    features, labels, image_paths = feature_extractor.extract_features_from_directory(
        data_dir)

    print(f"Extracted features from {len(features)} images")
    print(f"Feature vector size: {features.shape[1]}")
    print(f"Object types found: {set(labels)}")

    # PCA
    print('Performing PCA...')
    features_2d = cluster_functions.pca_reduce(features, 42)
    # Plot the results
    cluster_functions.plot_labels_2d('PCA', 'ml_plots', features_2d, labels, save=True)
    cluster_functions.plot_objects_2d('PCA', 'ml_plots', features_2d, labels, save=True)

    # t-SNE
    print('Performing t-SNE...')
    features_2d = cluster_functions.perform_tsne(features, 42)
    # Plot the results
    cluster_functions.plot_labels_2d('t-SNE', 'ml_plots', features_2d, labels, save=True)
    cluster_functions.plot_objects_2d('t-SNE', 'ml_plots', features_2d, labels, save=True)

    # # Apply t-SNE and visualize
    # tsne_results = visualize_tsne_clusters(features, labels)

    # # Analyze cluster separation
    # analyze_cluster_separation(tsne_results, labels)

    # # Save the plot
    # plt.savefig('tactile_tsne_clusters.png', dpi=300, bbox_inches='tight')
    # print("Visualization saved as 'tactile_tsne_clusters.png'")

    # # Save results to CSV for further analysis
    # results_df = pd.DataFrame({
    #     'image_path': image_paths,
    #     'object_type': labels,
    #     'tsne_x': tsne_results[:, 0],
    #     'tsne_y': tsne_results[:, 1]
    # })
    # results_df.to_csv('tsne_results.csv', index=False)
    # print("Results saved as 'tsne_results.csv'")

    # plt.show()


if __name__ == "__main__":
    main()

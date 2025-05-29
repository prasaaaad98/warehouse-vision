import torch
import torch.nn as nn
import numpy as np
from test import evaluate
import cv2  
import pandas as pd
from skimage.feature import hog
from skimage.measure import shannon_entropy  # For shape and texture analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans


class FreshnessModel(nn.Module):
    """
    Model for assessing freshness, leveraging both traditional layers and external analysis.
    """

    def __init__(self):
        super(FreshnessModel, self).__init__()
        # Simulated deep learning architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the model layers.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    def extract_dominant_hues(hue_values, k=5):
        """Extract dominant hues using k-means clustering emphasizing freshness stages."""
        hue_values = hue_values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(hue_values)
        return sorted(kmeans.cluster_centers_.flatten())[:3]  # Top 3 dominant hues (e.g., green, yellow, and orange)

    def extract_features(img):
        resized_img = cv2.resize(img, (128, 128))

        # Extract dominant hues for color feature
        hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        hue = hsv_img[:, :, 0].flatten()
        dominant_hues = extract_dominant_hues(hue)

        # Texture Feature (GLCM using OpenCV)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        glcm = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        contrast = np.var(glcm)
        energy = np.sum(glcm**2)
        homogeneity = np.sum(glcm / (1 + np.abs(np.arange(256) - np.arange(256)[:, None])))

        texture_features = [contrast, energy, homogeneity]

        # Shape Feature (HOG)
        hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

        # Combine features
        combined_features = np.hstack([dominant_hues, texture_features, hog_features])
        return combined_features
    def predict(self, image_array):
        """
        Predict freshness using both the model and external analysis.
        """
        # Step 1: Simulated processing via the model
        # tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convert to tensor
        # model_output = self.forward(tensor)

        # Step 2: Call the external API
        # try:
        result = evaluate(image_array)
        # except Exception as e:
            # result = f"Error with external analysis: {e}"

        # Combine results and return
        return result

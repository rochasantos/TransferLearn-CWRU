import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def clustering(model, dataloader, n_clusters, device="cuda"):
    """
    Applies K-means clustering to the features extracted by the trained model in Stage 1.

    Args:
        model (torch.nn.Module): Trained model from Stage 1.
        dataloader (torch.utils.data.DataLoader): DataLoader for labeled dataset.
        n_clusters (int): Number of clusters (same as the number of classes).
        device (str): Device to perform computations ("cpu" or "cuda").

    Returns:
        cluster_centers: Cluster centers determined by K-means.
        features: Extracted features from the dataloader.
        labels: True labels of the data.
    """
    model.eval()
    model.to(device)

    features = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            feature, _ = model(data)  # Extract features
            features.append(feature.cpu().numpy())
            labels.append(label.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters,n_init='auto', random_state=42)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_

    return cluster_centers, features, labels

def plot_clusters(features, labels, cluster_centers, n_clusters):
    """
    Visualizes the clusters using PCA for dimensionality reduction.

    Args:
        features (np.ndarray): Extracted features.
        labels (np.ndarray): True labels of the data.
        cluster_centers (np.ndarray): Cluster centers from K-means.
        n_clusters (int): Number of clusters.
    """
    # Reduce dimensions to 2 using PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    reduced_centers = pca.transform(cluster_centers)

    # Plot the features
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        cluster_points = reduced_features[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Class {i}", alpha=0.6)
    
    # Plot the cluster centers
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color="red", marker="x", s=100, label="Cluster Centers")

    plt.title("Cluster Visualization in Stage 2")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid()
    plt.show()

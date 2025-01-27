import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models import SignalFeatureCNN1D
from src.analysis import clustering, plot_clusters
from functools import partial
import random


# from utils.download_rawfile import download_rawfile
from utils import load_yaml
from src.data_processing import PtDataset, TransformPipeline


# DOWNLOAD RAW FILES
# def download():
#     for dataset in ["Paderborn"]:
#         download_rawfile(dataset)

    
# Define triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        # Triplet loss formula
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Step 2: K-means clustering algorithm
def apply_kmeans(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(features)
    return kmeans

# Step 3: Adjust with unlabeled data and KL-divergence
def kl_divergence_loss(unlabeled_features, cluster_centers):
    distances = torch.cdist(unlabeled_features, cluster_centers, p=2)
    soft_assignments = F.softmax(-distances, dim=1)
    hard_assignments = torch.argmax(soft_assignments, dim=1)
    loss = 0
    for i in range(len(cluster_centers)):
        assigned_points = unlabeled_features[hard_assignments == i]
        if len(assigned_points) > 0:
            mean_point = torch.mean(assigned_points, dim=0)
            loss += F.kl_div(mean_point.log(), cluster_centers[i], reduction='batchmean')
    return loss

if __name__ == '__main__':
    # download()
    dataset_name = "CWRU"
    sample_size = 120000
    segment_length = 140
    label_mapping = {"N": 0, "I": 1, "O": 2, "B": 3}
    n_class = len(label_mapping)
    n_epochs = 10 
    
    data_filter = load_yaml('config/filters_config.yaml')[dataset_name]
    dataset = PtDataset(data_filter, sample_size, apply_augmentation=True)   
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizers, and losses
    model = SignalFeatureCNN1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cross_entropy_loss = nn.CrossEntropyLoss()
    triplet_loss_fn = TripletLoss(margin=1.0)
    
    # Training loop
    for epoch in range(n_epochs):  # Number of epochs
        total_loss = 0
        for batch_data, batch_labels in data_loader:

            # Forward pass
            features, outputs = model(batch_data)

            # Cross-entropy loss
            ce_loss = cross_entropy_loss(outputs, batch_labels)

            # Triplet loss
            anchor = features[::3]  # Every third example as anchor
            positive = features[1::3]  # Every third example + 1 as positive
            negative = features[2::3]  # Every third example + 2 as negative

            # Ensure the triplet sizes match
            min_triplet_size = min(len(anchor), len(positive), len(negative))
            anchor = anchor[:min_triplet_size]
            positive = positive[:min_triplet_size]
            negative = negative[:min_triplet_size]

            triplet_loss = triplet_loss_fn(anchor, positive, negative)

            # Combine losses
            loss = ce_loss + triplet_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Total Loss: {total_loss:.4f}")

    # Clustering
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_clusters = len(label_mapping)
    cluster_centers, features, labels = clustering(model, data_loader, n_clusters, device=device)
    plot_clusters(features, labels, cluster_centers, n_clusters)


    # # Step 1: Extract features from labeled data
    # labeled_features = []
    # labeled_labels = []
    # for batch, labels in data_loader:
    #     with torch.no_grad():
    #         features = model(batch)
    #     labeled_features.append(features)
    #     labeled_labels.append(labels)
    # labeled_features = torch.cat(labeled_features)
    # labeled_labels = torch.cat(labeled_labels)

    # # Step 2: Apply K-means on labeled data
    # kmeans = apply_kmeans(labeled_features.numpy(), len(label_mapping))
    # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # # Step 3: Adjust with unlabeled data
    # unlabeled_data = torch.randn(50, 1, 160)  # Simulated unlabeled data
    # unlabeled_features = model(unlabeled_data)

    # Calculate KL-divergence loss
    # loss = kl_divergence_loss(labeled_features, cluster_centers)
    # print("KL-divergence loss:", loss.item())
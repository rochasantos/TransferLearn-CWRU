import os
from src.data_processing import DatasetManager
from torch.utils.data import DataLoader, ConcatDataset

annot = DatasetManager()

# Map feature values to class labels (N, B, O, I)
class_label_mapping = {
    "N": "Normal",
    "B": "Ball Fault",
    "O": "Outer Race Fault",
    "I": "Inner Race Fault"
}

# Compute and print class distribution for train and test splits
def grouper_distribution(dataset, feature_mitigation, indices, class_names):
    """
    Computes class distribution using the grouper logic.

    Args:
        dataset: The dataset object (supports ConcatDataset and standard datasets).
        feature_mitigation (str): Feature to group by (optional).
        indices (list): List of indices in the dataset to compute distribution.
        class_names (list): List of class names corresponding to label indices.

    Returns:
        dict: A dictionary with class names as keys and counts as values.
    """
    if not feature_mitigation:
        print('Group by: none')
        feature_mitigation = "label"  # Default to label if no feature is specified

    # Extract samples from ConcatDataset or standard dataset
    if isinstance(dataset, ConcatDataset):
        samples = []
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Initialize class distribution dictionary
    class_distribution = {class_name: 0 for class_name in class_names}

    # Traverse the specified indices to calculate distribution
    for idx in indices:
        path = samples[idx][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        feature_value = bearing_info.get(feature_mitigation, "default")
        class_label = bearing_info.get("label", "default")

        # Increment class count if the feature value is in class names
        if feature_value in class_names:
            class_distribution[feature_value] += 1

    # Print the computed class distribution
    print(f'Computed Class Distribution by {feature_mitigation}:', class_distribution)
    return class_distribution

def get_class_counter(dataset, feature_label="label"):
    """
    Counts the number of samples per class in the dataset.
    Supports both standard datasets and ConcatDataset.
    """
    class_counter = {"N": 0, "B": 0, "O": 0, "I": 0}

    # Handle both standard datasets and ConcatDataset
    if isinstance(dataset, ConcatDataset):
        # Combine samples from all constituent datasets
        samples = []
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Iterate through samples and count class occurrences
    for i in range(len(samples)):
        basename = os.path.basename(samples[i][0]).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        class_label = bearing_info.get(feature_label, "default")

        if class_label in class_counter:
            class_counter[class_label] += 1
        else:
            # Handle unexpected labels if necessary
            class_counter[class_label] = 1

    # Print the total samples per class
    print("Total samples per class:")
    for class_key, count in class_counter.items():
        print(f"  - {class_key}: {count}")

    return class_counter


def get_counter(dataset, feature_mitigation):
    """
    Calculates occurrences of a specific feature in the dataset.
    Handles both simple datasets and ConcatDataset.
    """
    counter = {}

    # Check if the dataset is a ConcatDataset
    if isinstance(dataset, ConcatDataset):
        # Combine samples from all constituent datasets
        samples = []
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Iterate through samples and count feature occurrences
    for i in range(len(samples)):
        basename = os.path.basename(samples[i][0]).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        feature_value = bearing_info.get(feature_mitigation, "default")

        if feature_value not in counter:
            counter[feature_value] = 0
        counter[feature_value] += 1

    # Print the class-wise counter
    print(f"Counter per class for feature '{feature_mitigation}':")
    for feature, count in counter.items():
        print(f"  - {feature}: {count}")

    return counter

def grouper(dataset, feature_mitigation):
    """
    Groups the dataset based on a specific feature for mitigation purposes.
    Supports both standard datasets and ConcatDataset.
    """
    if not feature_mitigation:
        print('Group by: none')
        # If `feature_mitigation` is empty, return a default group for all items
        return [0] * len(dataset)

    # Extract samples from ConcatDataset or standard dataset
    if isinstance(dataset, ConcatDataset):
        samples = []
        for sub_dataset in dataset.datasets:
            samples.extend(sub_dataset.samples)
    else:
        samples = dataset.samples

    # Initialize group metadata
    groups = []
    counter = get_counter(dataset, feature_mitigation)
    class_counter = get_class_counter(dataset, "label")
    num_groups = len(counter)

    # Initialize group class distribution tracker
    group_class_distribution = {g: {label: 0 for label in class_counter} for g in range(num_groups)}

    # Assign samples to groups
    for i in range(len(samples)):
        path = samples[i][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        feature_value = bearing_info.get(feature_mitigation, "default")
        class_label = bearing_info.get("label", "default")

        # Assign sample to an appropriate group
        assigned = False
        for group, distribution in group_class_distribution.items():
            if distribution[class_label] < class_counter[class_label] // num_groups:
                group_class_distribution[group][class_label] += 1
                groups.append(group)
                assigned = True
                break

        if not assigned:
            # Assign to a random group if balancing fails
            random_group = len(groups) % num_groups
            group_class_distribution[random_group][class_label] += 1
            groups.append(random_group)

    # Print detailed group information
    print('Group by:', feature_mitigation)
    print('Groups:', set(groups))
    print('Counter:', counter)
    print('ClassCounter:', class_counter)
    print('Group Class Distribution:')
    for group, distribution in group_class_distribution.items():
        print(f"  Group {group}: {distribution}")

    return groups

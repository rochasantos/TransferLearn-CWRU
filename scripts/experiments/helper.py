import os
from src.data_processing import DatasetManager

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
        dataset: The dataset object containing samples and metadata.
        feature_mitigation (str): Feature to group by (not used for this function).
        indices (list): List of indices in the dataset to compute distribution.
        class_names (list): List of class names corresponding to label indices.
    Returns:
        dict: A dictionary with class names as keys and counts as values.
    """
    if not feature_mitigation:
        print('Group by: none')
    
    # Initialize class distribution
    class_distribution = {class_name: 0 for class_name in class_names}

    for idx in indices:
        path = dataset.samples[idx][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        class_label = bearing_info.get("label", "default")
        
        # Increment class count if valid
        if class_label in class_names:
            class_distribution[class_label] += 1

    print('Computed Class Distribution:', class_distribution)
    return class_distribution

def get_class_counter(dataset, feature_label="class"):
    class_counter = {"N": 0, "B": 0, "O": 0, "I": 0}
    for i in range(len(dataset)):
        basename = os.path.basename(dataset.samples[i][0]).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        class_label = bearing_info.get("label", "default")

        if class_label in class_counter:
            class_counter[class_label] += 1
        else:
            # Handle unexpected labels if necessary
            class_counter[class_label] = 1

    # Print the total samples per class
    # print("Total samples per class:")
    # for class_key, count in class_counter.items():
    #     class_name = class_label_mapping.get(class_key, class_key)
    #     print(f"  - {class_name} ({class_key}): {count}")

    return class_counter


def get_counter(dataset, feature_mitigation):
    counter = {}
    for i in range(len(dataset)):
        basename = os.path.basename(dataset.samples[i][0]).split('#')[0]
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
    if not feature_mitigation:
        print('Group by: none')
        # If `feature_mitigation` is empty, return a default group for all items
        return [0] * len(dataset)
    
    groups = []
    hash = {}
    counter = get_counter(dataset, feature_mitigation)
    class_counter = get_class_counter(dataset, "class")
    group_class_distribution = {g: {label: 0 for label in class_counter} for g in range(len(counter))}

    # Assign groups
    for i in range(len(dataset)):
        path = dataset.samples[i][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_data({"filename": basename})[0]
        feature_value = bearing_info.get(feature_mitigation, "default")
        class_label = bearing_info.get("label", "default")

        # Ensure all classes are distributed across groups
        assigned = False
        for group, distribution in group_class_distribution.items():
            if distribution[class_label] < class_counter[class_label] // len(group_class_distribution):
                group_class_distribution[group][class_label] += 1
                groups.append(group)
                assigned = True
                break

        if not assigned:
            # Assign to a random group if balancing fails
            random_group = len(groups) % len(group_class_distribution)
            group_class_distribution[random_group][class_label] += 1
            groups.append(random_group)

    print('Group by:', feature_mitigation)
    print('Groups:', set(groups))
    print('Counter:', counter)
    print('ClassCounter:', class_counter)
    print('Group Class Distribution:')
    for group, distribution in group_class_distribution.items():
        print(f"  Group {group}: {distribution}")
        
    return groups

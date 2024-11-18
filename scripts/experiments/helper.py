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

def get_class_counter(dataset, feature_label="class"):
    class_counter = {"N": 0, "B": 0, "O": 0, "I": 0}
    for i in range(len(dataset)):
        basename = os.path.basename(dataset.samples[i][0]).split('#')[0]
        bearing_info = annot.filter_dataset(filename=fr'\b{basename}\b')[0]
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
        bearing_info = annot.filter_dataset(filename=fr'\b{basename}\b')[0]
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
        bearing_info = annot.filter_dataset(filename=basename)[0]
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
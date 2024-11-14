import os
from src.data_processing import DatasetManager

annot = DatasetManager()

def get_counter(dataset, feature_mitigation):
    counter = {}
    for i in range(len(dataset)):
        basename = os.path.basename(dataset.samples[i][0]).split('#')[0]
        bearing_info = annot.filter_dataset(filename=fr'\b{basename}\b')[0]
        feature_value = bearing_info.get(feature_mitigation, "default")

        if feature_value not in counter:
            counter[feature_value] = 0
        counter[feature_value] += 1

    return counter

def grouper(dataset, feature_mitigation):
    if not feature_mitigation:
        print('Group by: none')
        # If `feature_mitigation` is empty, return a default group for all items
        return [0] * len(dataset)
    
    groups = []
    hash = {}
    counter = get_counter(dataset, feature_mitigation)
    c = 0

    for i in range(len(dataset)):
        path = dataset.samples[i][0]
        basename = os.path.basename(path).split('#')[0]
        bearing_info = annot.filter_dataset(filename=basename)[0]
        feature_value = bearing_info.get(feature_mitigation, "default")

        if bearing_info.get("extent_damage") == '000':
            groups.append(c)
            c = (c + 1) % (len(counter) - 1)
        else:
            if feature_value not in hash:
                hash[feature_value] = len(hash)
            groups.append(hash[feature_value])

    print('Group by: ',feature_mitigation )
    print('Groups:', set(groups))
    print('Counter:', counter)
    print('Hash:', hash)

    return groups
import os
from src.data_processing import DatasetManager

metainfo = DatasetManager()

def get_counter(dataset, feature_mitigation):
    counter = {}
    for i in range(len(dataset)):
        basename = os.path.basename(dataset.samples[i][0]).split('#')[0]
        bearing_info = metainfo.filter_data({"filename": basename})[0]
        feature_value = bearing_info[feature_mitigation]

        if feature_value not in counter:
            counter[feature_value]=0
        counter[feature_value]+=1

    return counter

def grouper(dataset, feature_mitigation):
    groups=[]
    hash = dict()
    counter = get_counter(dataset, feature_mitigation)
    c=0
    for i in range(len(dataset)):
        path = dataset.samples[i][0]
        basename = os.path.basename(path)
        bearing_info = metainfo.filter_data({"filename": basename.split('#')[0]})[0]
        feature_value = bearing_info[feature_mitigation]
        # Distribution of reference bearing values.
        if bearing_info["extent_damage"]=='000':
            groups.append(c)
            c = c + 1 if c < len(counter)-2 else 0
        else:
            if feature_value not in hash:
                hash[feature_value] = len(hash)
            groups.append(hash[feature_value])


    print('Groups:', set(groups))
    print('Couter:', counter)
    print('Hash:', hash)

    return groups

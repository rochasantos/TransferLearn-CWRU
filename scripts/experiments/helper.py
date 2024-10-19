import os
from src.data_processing.annotation_file import AnnotationFileHandler

annot = AnnotationFileHandler()

def get_counter(dataset, feature_mitigation):
    counter = {}
    for i in range(len(dataset)):
        basename = os.path.basename(dataset.samples[i][0])
        bearing_info = annot.filter_data(filename=basename.split('#')[0])[0]
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
        basename = os.path.basename(dataset.samples[i][0])
        bearing_info = annot.filter_data(filename=basename.split('#')[0])[0]
        feature_value = bearing_info[feature_mitigation]
        # Distribution of reference bearing values.
        if bearing_info["extent_damage"]=='000':
            groups.append(c)
            if c == len(counter)-1:
                c=0
            else:
                c=+1
        else:
            if feature_value not in hash:
                hash[feature_value] = len(hash)
            groups.append(hash[feature_value])

    print('Groups:', set(groups))
    print('Couter:', counter)

    return groups
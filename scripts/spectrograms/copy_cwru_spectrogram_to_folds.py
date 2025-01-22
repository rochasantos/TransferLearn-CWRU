import os
from utils import copy_file
from src.data_processing.dataset_manager import DatasetManager


def copy_cwru_spectrogram_to_folds(directory = "data/spectrograms/cwru/", config='extent_damage',
                              labels = ["N", "I", "O", "B"], out_dir="cwru_cv", balanced=None):
    metainfo = DatasetManager("CWRU")

    list_file_removed = [
        '144', '145', '146', '147', '156', '158', '159', '160',
        '246', '247', '248', '249', '258', '259', '260', '261',
        '148', '149', '150', '151', '161', '162', '163', '164',
        '250', '251', '252', '253', '262', '263', '264', '265'
        ]

    # creating a map config x fold
    n_fold = 1
    map = {}
    n_fold_ref = 1
    map_ref = {}
    for info in metainfo.filter_data():
        config_value = info[config]
        extent_damage = info['extent_damage']
        motor_load = info['hp']
        if extent_damage == "000":
            if motor_load not in map_ref:
                map_ref[motor_load] = f"fold{n_fold_ref}"
                n_fold_ref += 1
        else: 
            if config_value not in map:
                map[config_value] = f"fold{n_fold}"
                n_fold += 1

    print('MAP:', map)
    print('MAP_REF:', map_ref)
    
    # print("Copying files from the 'data/spectrograms/cwru directory' to the folds in the 'data/spectrograms/cwru_cv' directory")
    
    directory = "data/spectrograms/cwru/"
    for root, dir, files in os.walk(directory):
        label = root[-1]
        if label in labels:
            for file in files:
                if not file.endswith('.png'):
                    continue
                basename = file.split("#")[0]
                if balanced:
                    if basename in list_file_removed:
                        continue
                info = metainfo.filter_data({"filename": basename})[0]
                assert len(info) !=1 , "Error trying to locate file in annotation_file file."
                config_value = info[config]
                
                fold_name = map_ref[info['hp']] if info['extent_damage'] == '000' else map[config_value]
                
                source_path = os.path.join(directory, label, file)
                out_dir = 'cwru_balanced_'+config if balanced else 'cwru_'+config
                destination_path = os.path.join("data/spectrograms/", out_dir, fold_name, label, file)
                if os.path.exists(destination_path):
                    continue
                destination_dir = os.path.dirname(destination_path)
                if not os.path.isdir(destination_dir):
                    os.makedirs(destination_dir, exist_ok=True)
                copy_file(source_path, destination_path)
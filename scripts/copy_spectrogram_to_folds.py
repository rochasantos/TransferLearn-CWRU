import os
from utils import copy_file, list_files
from src.data_processing.dataset_manager import DatasetManager

def extension_fold_mapping(basename):
    return {
    "000": "fold"+str(int(basename)-96),
    "007": "fold1", 
    "014": "fold2", 
    "021": "fold3", 
    "028": "fold4" }

def copy_spectrogram_to_folds():
    metainfo = DatasetManager("CWRU")
    
    root_dir = "data/spectrograms/cwru/"
    for label in ["N", "I", "O", "B"]:
        list_of_files = list_files(root_dir+label, extension=".png")

        for file in list_of_files:
            basename = file.split("#")[0]
            info = metainfo.filter_data({"filename": basename})[0]
            extent_damage = info["extent_damage"]
            fold = extension_fold_mapping(basename)[extent_damage]
            source_path = os.path.join(root_dir, label, file)
            destination_path = os.path.join("data/spectrograms/cwru_cv", fold, label, file)
            copy_file(source_path, destination_path)
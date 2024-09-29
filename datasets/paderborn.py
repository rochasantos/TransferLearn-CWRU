import os
from datasets.base_dataset import BaseDataset
from utils.download_extract import extract_rar, remove_rar_files

def get_list_of_bearings(n_acquisitions, bearing_names):
    settings_files = ["N15_M07_F10_", "N09_M07_F10_", "N15_M01_F10_", "N15_M07_F04_"]
    list_of_bearings = []
    for bearing in bearing_names:
        if bearing[1] == '0':
            tp = "Normal_"
        elif bearing[1] == 'A':
            tp = "OR_"
        else:
            tp = "IR_"
        for idx, setting in enumerate(settings_files):
            for i in range(1, n_acquisitions + 1):
                key = tp + bearing + "_" + str(idx) + "_" + str(i)
                list_of_bearings.append((key, os.path.join(bearing, setting + bearing +
                                                "_" + str(i) + ".mat")))
    return list_of_bearings

class Paderborn(BaseDataset):    
    
    def list_of_bearings(self):
        return [
        ("K001.rar", "K001.rar"), ("K002.rar", "K002.rar"), ("K003.rar", "K003.rar"), ("K004.rar", "K004.rar"), ("K005.rar", "K005.rar"), ("K006.rar", "K006.rar"), 
        ("KA01.rar", "KA01.rar"), ("KA03.rar", "KA03.rar"), ("KA04.rar", "KA04.rar"), ("KA05.rar", "KA05.rar"), ("KA06.rar", "KA06.rar"), ("KA07.rar", "KA07.rar"), ("KA09.rar", "KA09.rar"), ("KA15.rar", "KA15.rar"), ("KA16.rar", "KA16.rar"), ("KA22.rar", "KA22.rar"), ("KA30.rar", "KA30.rar"), 
        ("KI01.rar", "KI01.rar"), ("KI03.rar", "KI03.rar"), ("KI04.rar", "KI04.rar"), ("KI05.rar", "KI05.rar"), ("KI07.rar", "KI07.rar"), ("KI08.rar", "KI08.rar"), ("KI14.rar", "KI14.rar"), ("KI16.rar", "KI16.rar"), ("KI17.rar", "KI17.rar"), ("KI18.rar", "KI18.rar"), ("KI21.rar", "KI21.rar"), 
        ]

    def __init__(self):
        super().__init__(rawfilesdir = "data/raw/paderborn", 
                         url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/")

    def extract_rar(self, remove_rarfile=False):
        for bearing in self.list_of_bearings():
            rar_path = os.path.join(self.rawfilesdir, bearing[1])
            dirname = self.rawfilesdir
            if not os.path.isdir(os.path.splitext(rar_path)[0]):
                extract_rar(dirname, rar_path)
        if remove_rarfile:
            remove_rar_files(self.rawfilesdir)
    
    def __str__(self):
        return "Paderborn"
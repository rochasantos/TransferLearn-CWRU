import os
import numpy as np
import re
import scipy.io
from datasets.base_dataset import BaseDataset
from utils.download_extract import extract_rar, remove_rar_files

def _extract_label(filepath):
    tag = filepath.split('_')[-2]
    if tag == '0':
        tp = "N_"
    elif tag == 'A':
        tp = "O_"
    else:
        tp = "I_"
    
    filename = os.path.basename(filepath).split('.')[0]
    return tp+filename


class Paderborn(BaseDataset):    

    def __init__(self, debug=False):
        super().__init__(rawfilesdir = "data/raw/paderborn",
                         spectdir="data/processed/paderborn_spectrograms",
                         sample_rate=64000,
                         url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/",
                         debug=debug)

    def list_of_bearings(self):
        return [("K001.rar", "K001.rar")] if self.debug else [
        ("K001.rar", "K001.rar"), ("K002.rar", "K002.rar"), ("K003.rar", "K003.rar"), ("K004.rar", "K004.rar"), ("K005.rar", "K005.rar"), ("K006.rar", "K006.rar"), 
        ("KA01.rar", "KA01.rar"), ("KA03.rar", "KA03.rar"), ("KA04.rar", "KA04.rar"), ("KA05.rar", "KA05.rar"), ("KA06.rar", "KA06.rar"), ("KA07.rar", "KA07.rar"), ("KA09.rar", "KA09.rar"), ("KA15.rar", "KA15.rar"), ("KA16.rar", "KA16.rar"), ("KA22.rar", "KA22.rar"), ("KA30.rar", "KA30.rar"), 
        ("KI01.rar", "KI01.rar"), ("KI03.rar", "KI03.rar"), ("KI04.rar", "KI04.rar"), ("KI05.rar", "KI05.rar"), ("KI07.rar", "KI07.rar"), ("KI08.rar", "KI08.rar"), ("KI14.rar", "KI14.rar"), ("KI16.rar", "KI16.rar"), ("KI17.rar", "KI17.rar"), ("KI18.rar", "KI18.rar"), ("KI21.rar", "KI21.rar"), 
        ]
    
    def _extract_rar(self, remove_rarfile=False):
        for bearing in self.list_of_bearings():
            rar_path = os.path.join(self.rawfilesdir, bearing[1])
            dirname = self.rawfilesdir
            if not os.path.isdir(os.path.splitext(rar_path)[0]):
                extract_rar(dirname, rar_path)
        if remove_rarfile:
            remove_rar_files(self.rawfilesdir)

    def download(self):
        super().download()
        self._extract_rar(remove_rarfile=True)

    def load_signal(self, acquisition_maxsize=None, regex_filter=r'.*N09_M07_F10_K001_1\.mat$'):
        """
        Extracts data from a MATLAB .mat file for a specific accelerometer position.

        Parameters:
        - filepath (str): The path to the .mat file.
        - acquisition_maxsize (int): Maximum number of samples to extract (default: 12000).
        
        Returns:
        - np.array: Extracted accelerometer data.
        """
        
        regex = re.compile(regex_filter)

        signal = []
        label = []
        for root, dirs, files in os.walk(self.rawfilesdir):
            for file in files:
                filepath = os.path.join(root, file)
               
                if not regex.search(file):
                    continue

                matlab_file = scipy.io.loadmat(filepath)
                key = os.path.basename(filepath).split('.')[0]
                data_raw = matlab_file[key]['Y'][0][0][0][6][2][0, :]
                
                if acquisition_maxsize:
                    signal.append(data_raw[:acquisition_maxsize])
                else:
                    signal.append(data_raw)
                
                label.append(_extract_label(filepath))

        self._data = np.array(signal)
        self._label = np.array(label)
        

    def __str__(self):
        return "Paderborn"
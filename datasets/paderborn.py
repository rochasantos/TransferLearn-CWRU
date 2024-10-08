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
    """
    Paderborn Dataset Class

    This class manages the Paderborn bearing dataset used for fault diagnosis.
    It provides methods for listing bearing files, loading vibration signals, and setting up dataset attributes.
    This class inherits from BaseDataset the load_signal methods responsible for loading and downloading data.
    
    Attributes
        rawfilesdir (str) : Directory where raw data files are stored.
        spectdir (str) : Directory where processed spectrograms will be saved.
        sample_rate (int) : Sampling rate of the vibration data.
        url (str) : URL for downloading the Paderborn dataset.
        debug (bool) : If True, limits the number of files processed for faster testing.

    Methods
        list_of_bearings(): Returns a list of tuples with filenames and URL suffixes for downloading vibration data.
        download(): Downloads .mat files from dataset website URL, and extracts them from .rar files. 
        _extract_rar(): Extracts the .mat files from .rar files.
        _extract_data(): Extracts the vibration signal data from .mat files.
        __str__(): Returns a string representation of the dataset.
    """

    def __init__(self, debug=False):
        super().__init__(rawfilesdir = "data/raw/paderborn",
                         spectdir="data/processed/paderborn_spectrograms",
                         sample_rate=64000,
                         url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/",
                         debug=debug)

    def list_of_bearings(self):
        """ 
        Returns: 
            A list of tuples containing filenames (for naming downloaded files) and URL suffixes 
            for downloading vibration data.
        """
        return [("K001.rar", "K001.rar")] if self.debug else [
        ("K001.rar", "K001.rar"), ("K002.rar", "K002.rar"), ("K003.rar", "K003.rar"), ("K004.rar", "K004.rar"), ("K005.rar", "K005.rar"), ("K006.rar", "K006.rar"), 
        ("KA01.rar", "KA01.rar"), ("KA03.rar", "KA03.rar"), ("KA04.rar", "KA04.rar"), ("KA05.rar", "KA05.rar"), ("KA06.rar", "KA06.rar"), ("KA07.rar", "KA07.rar"), ("KA09.rar", "KA09.rar"), ("KA15.rar", "KA15.rar"), ("KA16.rar", "KA16.rar"), ("KA22.rar", "KA22.rar"), ("KA30.rar", "KA30.rar"), 
        ("KI01.rar", "KI01.rar"), ("KI03.rar", "KI03.rar"), ("KI04.rar", "KI04.rar"), ("KI05.rar", "KI05.rar"), ("KI07.rar", "KI07.rar"), ("KI08.rar", "KI08.rar"), ("KI14.rar", "KI14.rar"), ("KI16.rar", "KI16.rar"), ("KI17.rar", "KI17.rar"), ("KI18.rar", "KI18.rar"), ("KI21.rar", "KI21.rar"), 
        ]
    
    def _extract_rar(self, remove_rarfile=False):
        """ Extracts .mat files from .rar files and removes them if remove_rarfile is True.
        Args:
            remove_rarfile (bool): Allows to remove .rar files after extraction.
        """     
        for bearing in self.list_of_bearings():
            rar_path = os.path.join(self.rawfilesdir, bearing[1])
            dirname = self.rawfilesdir
            if not os.path.isdir(os.path.splitext(rar_path)[0]):
                extract_rar(dirname, rar_path)
        if remove_rarfile:
            remove_rar_files(self.rawfilesdir)

    def download(self):
        """ Downloads and extracts .mat files from .rar files.
        """
        super().download()
        self._extract_rar(remove_rarfile=True)

    def _extract_data(self, filepath):
        """ Extracts data from a .mat file for bearing fault analysis.
        Args:
            filepath (str): The path to the .mat file.
        Return:
            tuple: A tuple containing the extracted data and its label.
        """           
        matlab_file = scipy.io.loadmat(filepath)
        key = os.path.basename(filepath).split('.')[0]
        data_raw = matlab_file[key]['Y'][0][0][0][6][2][0, :]
        label = _extract_label(filepath)
        if self.acquisition_maxsize:
            return data_raw[:self.acquisition_maxsize], label
        else:
            return data_raw, label        

    def __str__(self):
        return "Paderborn"
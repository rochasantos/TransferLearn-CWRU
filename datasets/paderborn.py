import os
import scipy.io
from datasets.base_dataset import BaseDataset
from src.data_processing import DatasetManager
from utils.download_extract import extract_rar

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

    def __init__(self):
        
        super().__init__(rawfilesdir = "data/raw/paderborn",
                         url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/")
        
        self.all_files_metadata = DatasetManager().filter_data(dataset_name='Paderborn')


    def list_of_bearings(self):
        """ 
        Returns: 
            A list of tuples containing filenames (for naming downloaded files) and URL suffixes 
            for downloading vibration data.
        """
        return [
        ("K001", "K001.rar"), ("K002", "K002.rar"), ("K003", "K003.rar"), ("K004", "K004.rar"), ("K005", "K005.rar"), ("K006", "K006.rar"), 
        ("KA01", "KA01.rar"), ("KA03", "KA03.rar"), ("KA04", "KA04.rar"), ("KA05", "KA05.rar"), ("KA06", "KA06.rar"), ("KA07", "KA07.rar"), ("KA09", "KA09.rar"), ("KA15", "KA15.rar"), ("KA16", "KA16.rar"), ("KA22", "KA22.rar"), ("KA30", "KA30.rar"), 
        ("KI01", "KI01.rar"), ("KI03", "KI03.rar"), ("KI04", "KI04.rar"), ("KI05", "KI05.rar"), ("KI07", "KI07.rar"), ("KI08", "KI08.rar"), ("KI14", "KI14.rar"), ("KI16", "KI16.rar"), ("KI17", "KI17.rar"), ("KI18", "KI18.rar"), ("KI21", "KI21.rar"), 
        ]
    
    
    def _extract_rar(self):
        """ Extracts .mat files from .rar files and removes them if remove_rarfile is True.
        """
        for bearing in self.list_of_bearings():
            rar_path = os.path.join(self.rawfilesdir, bearing[1])
            extract_rar(rar_path, self.rawfilesdir)


    def download(self):
        """ Downloads and extracts .mat files from .rar files.
        """
        super().download()
        self._extract_rar()


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
        file_metadata = list(filter(lambda x: x["filename"]==key, self.all_files_metadata))[0]
        label = file_metadata['label']
        if self.acquisition_maxsize:
            return data_raw[:self.acquisition_maxsize], label
        else:
            return data_raw, label        

    def __str__(self):
        return "Paderborn"
    
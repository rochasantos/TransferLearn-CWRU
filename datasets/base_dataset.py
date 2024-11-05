import os
import numpy as np
import re
from utils.download_extract import download_file
from src.data_processing import DatasetManager

from abc import ABC, abstractmethod

class BaseDataset(ABC):
    
    def __init__(self, rawfilesdir, url):
        """
        Base class for all dataset models. 
        Defines the attributes and the download and load_signal functions, 
        along with an abstract extract_data method. The extract_data method 
        delegates the responsibility of data extraction to the subclasses, 
        requiring them to implement their specific extraction logic.

        Parameters:
        - rawfilesdir (str): The directory where raw files will be stored.
        - url (str): The base URL for downloading the dataset files.
        
        Methods:
            download(): Downloads .mat from the dataset website URL.
        """
        self._rawfilesdir = rawfilesdir  # Directory to store the raw files
        self._url = url  # Base URL for downloading the files
        self._data = [] # List to store the extracted data.
        self._label = []  # List to store the corresponding labels for the data.
        self.acquisition_maxsize = None  # Maximum size for data acquisition.
        self._annotation_file=DatasetManager().filter_dataset(dataset_name=self.__class__.__name__)

        if not os.path.exists(self._rawfilesdir):
            os.makedirs(self._rawfilesdir)

    def list_of_bearings(self):
        pass

    def download(self):
        """ Download files from datasets website.
        """
        url = self._url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print(f"Stating download of {self} dataset.")
        list_of_bearings = self.list_of_bearings()
        dataset_name = self.__class__.__name__.lower()
        unit = '.mat'
        if dataset_name == "paderborn":
            unit = '.rar'
        for bearing in list_of_bearings:
            sufix_url = bearing[1]
            output_path = os.path.join('data/raw', dataset_name, bearing[0]+unit)
            if not os.path.exists(os.path.join(dirname, sufix_url)):
                download_file(url, sufix_url, output_path)                
        print("Download finished.")

    def load_signal_by_path(self, filepath):
        signal, label = self._extract_data(filepath)
        return signal, label
    
    def load_signal(self, regex_filter=r'.*\.mat$'):
        """ Load vibration signal data from .mat files, filtered by a regex. 
        Args:
            regex_filter (str): Regular expression to filter filenames.
        Returns:
            None
        """
        regex = re.compile(regex_filter)
        signal = []
        labels = []

        for root, dirs, files in os.walk(self.rawfilesdir):
            for file in files:
                filepath = os.path.join(root, file)
                if not regex.search(file):
                    continue
                data, label = self._extract_data(filepath)
                signal.append(data)
                labels.append(label)

        min_size_acquisition = min([np.size(data) for data in signal])
        trimmed_data = [data[:min_size_acquisition] for data in signal]

        self._data = np.array(trimmed_data)
        self._label = np.array(labels)

    @classmethod
    @abstractmethod
    def _extract_data(self, filepath):
        """ This method is responsible for extracting data from a bearing fault dataset in a .mat file.
        Returns:
            tuple: A tuple containing (data, label), where 'data' is the extracted dataset and 'label' is the corresponding label.
        """
        pass  

    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label
    
    @property
    def rawfilesdir(self):
        return self._rawfilesdir
        
    @property
    def annotation_file(self):
        return self._annotation_file
import os
import numpy as np
import re
import pandas as pd
import csv
from utils.download_extract import download_file
from src.data_processing.annotation_file import AnnotationFileHandler

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
        self._annotation_file=AnnotationFileHandler().filter_data(dataset_name=self.__class__.__name__)


        if not os.path.exists(self._rawfilesdir):
            os.makedirs(self._rawfilesdir)   

    def download(self):
        """ Download files from datasets website.
        """
        url = self.url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print(f"Stating download of {self} dataset.")
        filename_list = [info["filename"]+'.mat' for info in self.annotation_file]
        for filename in filename_list:
            if not os.path.exists(os.path.join(dirname, filename)):
                download_file(url, filename, dirname, filename)
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
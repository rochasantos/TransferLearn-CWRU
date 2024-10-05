import os
import numpy as np
import re
import scipy.io
from utils.download_extract import download_file

class BaseDataset:
    
    def __init__(self, rawfilesdir, spectdir, sample_rate, url):
        """
        Base class for all datasets. 
        Defines attributes and a download function to be inherited by specific datasets.

        Parameters:
        - rawfilesdir (str): The directory where raw files will be stored.
        - url (str): The base URL for downloading the dataset files.
        """
        self._rawfilesdir = rawfilesdir  # Directory to store the raw files
        self._spectdir = spectdir  # Directory to store the raw files
        self._sample_rate = sample_rate  # Directory to store the raw files
        self._url = url  # Base URL for downloading the files
        self._data = []
        self._label = []

        # Check if the raw files directory exists, if not, create it
        if not os.path.exists(self._rawfilesdir):
            os.makedirs(self._rawfilesdir)

    
    def download(self):
        """
        Download files from datasets website.
        """
        url = self.url
        dirname = self.rawfilesdir
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        print(f"Stating download of {self} dataset.")
        bearings = self.list_of_bearings()
        for filename, url_suffix in bearings:
            download_file(url, dirname, url_suffix, filename)
        print("Download finished.")

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
    def spectdir(self):
        return self._spectdir
    
    @property
    def sample_rate(self):
        return self._sample_rate
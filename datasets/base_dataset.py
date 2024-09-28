import os
from utils.download_extract import download_file

class BaseDataset:
    
    def __init__(self, rawfilesdir, url):
        """
        Base class for all datasets. 
        Defines attributes and a download function to be inherited by specific datasets.

        Parameters:
        - rawfilesdir (str): The directory where raw files will be stored.
        - url (str): The base URL for downloading the dataset files.
        """
        self.rawfilesdir = rawfilesdir  # Directory to store the raw files
        self.url = url  # Base URL for downloading the files

        # Check if the raw files directory exists, if not, create it
        if not os.path.exists(self.rawfilesdir):
            os.makedirs(self.rawfilesdir)

    
    def download(self):
        """
        Download files from CWRU website.
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
  

import sys
import os
p_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(p_root)

import scipy.io
import numpy as np
import os
import re
from datasets.base_dataset import BaseDataset

class CWRU(BaseDataset):
    """
    CWRU Dataset Class

    This class manages the CWRU (Case Western Reserve University) bearing dataset used for fault diagnosis.
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
        _extract_data(): Extracts the vibration signal data from .mat files.
        __str__(): Returns a string representation of the dataset.
    """

    def __init__(self):

        super().__init__(rawfilesdir = "data/raw/cwru",
                         url = "https://engineering.case.edu/sites/default/files/")

    def _extract_data(self, filepath):
        """ Extracts data from a .mat file for bearing fault analysis.
        Args:
            filepath (str): The path to the .mat file.
        Return:
            tuple: A tuple containing the extracted data and its label.
        """
        matlab_file = scipy.io.loadmat(filepath)
        keys = re.findall(r'X\d{3}_[A-Z]{2}_time', str(matlab_file.keys()))
        map_position = {
            '6203': 'FE',
            '6205': 'DE' }
        basename = os.path.basename(filepath).split('.')[0]
        annot_info = list(filter(lambda x: x["filename"]==basename, self.annotation_file))[0]
        label = annot_info["label"]
        bearing_type = annot_info["bearing_type"]
        bearing_position = ['DE'] if label == 'N' else [map_position[bearing_type]]
        for key in keys:
            if key[-7:-5] in bearing_position:
                data_squeezed = np.squeeze(matlab_file[key])  # removes the dimension corresponding to 
                                                              # the number of channels, as only a single channel is being used.
                if self.acquisition_maxsize:
                    return data_squeezed[:self.acquisition_maxsize], label
                else:
                    return data_squeezed, label

    def __str__(self):
        return "CWRU"
    
    
if __name__ == '__main__':
    dataset = CWRU()
    dataset.download()
    # dataset.load_signal(r'[10]{3}\.mat')
    # data, label = dataset.data, dataset.label
    # print(data.shape, label)

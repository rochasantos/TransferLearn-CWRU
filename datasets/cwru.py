import scipy.io
import numpy as np
import os
import re
from datasets.base_dataset import BaseDataset


class CWRU(BaseDataset):    
    
    def list_of_bearings(self):
        if self.debug:
            return [
            ("N.000.NN_0&12000.mat","97.mat"), ("I.007.DE_0&48000.mat","109.mat"), ("B.007.DE_0&48000.mat","122.mat"), ("O.007.DE.@6_0&48000.mat","135.mat"),
            ("I.007.DE_0&12000.mat","105.mat"), ("B.007.DE_0&12000.mat","118.mat"), ("O.007.DE.@6_0&12000.mat","130.mat")] 
        else:
            return [
            ("N.000.NN_0&12000.mat","97.mat"),        ("N.000.NN_1&12000.mat","98.mat"),        ("N.000.NN_2&12000.mat","99.mat"),        ("N.000.NN_3&12000.mat","100.mat"),
            ("I.007.DE_0&48000.mat","109.mat"),       ("I.007.DE_1&48000.mat","110.mat"),       ("I.007.DE_2&48000.mat","111.mat"),       ("I.007.DE_3&48000.mat","112.mat"),
            ("B.007.DE_0&48000.mat","122.mat"),       ("B.007.DE_1&48000.mat","123.mat"),       ("B.007.DE_2&48000.mat","124.mat"),       ("B.007.DE_3&48000.mat","125.mat"),    
            ("O.007.DE.@6_0&48000.mat","135.mat"),    ("O.007.DE.@6_1&48000.mat","136.mat"),    ("O.007.DE.@6_2&48000.mat","137.mat"),    ("O.007.DE.@6_3&48000.mat","138.mat"),    
            ("O.007.DE.@3_0&48000.mat","148.mat"),    ("O.007.DE.@3_1&48000.mat","149.mat"),    ("O.007.DE.@3_2&48000.mat","150.mat"),    ("O.007.DE.@3_3&48000.mat","151.mat"),    
            ("O.007.DE.@12_0&48000.mat","161.mat"),   ("O.007.DE.@12_1&48000.mat","162.mat"),   ("O.007.DE.@12_2&48000.mat","163.mat"),   ("O.007.DE.@12_3&48000.mat","164.mat"),    
            ("I.014.DE_0&48000.mat","174.mat"),       ("I.014.DE_1&48000.mat","175.mat"),       ("I.014.DE_2&48000.mat","176.mat"),       ("I.014.DE_3&48000.mat","177.mat"),    
            ("B.014.DE_0&48000.mat","189.mat"),       ("B.014.DE_1&48000.mat","190.mat"),       ("B.014.DE_2&48000.mat","191.mat"),       ("B.014.DE_3&48000.mat","192.mat"),    
            ("O.014.DE.@6_0&48000.mat","201.mat"),    ("O.014.DE.@6_1&48000.mat","202.mat"),    ("O.014.DE.@6_2&48000.mat","203.mat"),    ("O.014.DE.@6_3&48000.mat","204.mat"),    
            ("I.021.DE_0&48000.mat","213.mat"),       ("I.021.DE_1&48000.mat","214.mat"),       ("I.021.DE_2&48000.mat","215.mat"),       ("I.021.DE_3&48000.mat","217.mat"),    
            ("B.021.DE_0&48000.mat","226.mat"),       ("B.021.DE_1&48000.mat","227.mat"),       ("B.021.DE_2&48000.mat","228.mat"),       ("B.021.DE_3&48000.mat","229.mat"),    
            ("O.021.DE.@6_0&48000.mat","238.mat"),    ("O.021.DE.@6_1&48000.mat","239.mat"),    ("O.021.DE.@6_2&48000.mat","240.mat"),    ("O.021.DE.@6_3&48000.mat","241.mat"),    
            ("O.021.DE.@3_0&48000.mat","250.mat"),    ("O.021.DE.@3_1&48000.mat","251.mat"),    ("O.021.DE.@3_2&48000.mat","252.mat"),    ("O.021.DE.@3_3&48000.mat","253.mat"),    
            ("O.021.DE.@12_0&48000.mat","262.mat"),   ("O.021.DE.@12_1&48000.mat","263.mat"),   ("O.021.DE.@12_2&48000.mat","264.mat"),   ("O.021.DE.@12_3&48000.mat","265.mat"),    
            ("I.007.DE_0&12000.mat","105.mat"),       ("I.007.DE_1&12000.mat","106.mat"),       ("I.007.DE_2&12000.mat","107.mat"),       ("I.007.DE_3&12000.mat","108.mat"),
            ("B.007.DE_0&12000.mat","118.mat"),       ("B.007.DE_1&12000.mat","119.mat"),       ("B.007.DE_2&12000.mat","120.mat"),       ("B.007.DE_3&12000.mat","121.mat"),    
            ("O.007.DE.@6_0&12000.mat","130.mat"),    ("O.007.DE.@6_1&12000.mat","131.mat"),    ("O.007.DE.@6_2&12000.mat","132.mat"),    ("O.007.DE.@6_3&12000.mat","133.mat"),    
            ("O.007.DE.@3_0&12000.mat","144.mat"),    ("O.007.DE.@3_1&12000.mat","145.mat"),    ("O.007.DE.@3_2&12000.mat","146.mat"),    ("O.007.DE.@3_3&12000.mat","147.mat"),    
            ("O.007.DE.@12_0&12000.mat","156.mat"),   ("O.007.DE.@12_1&12000.mat","158.mat"),   ("O.007.DE.@12_2&12000.mat","159.mat"),   ("O.007.DE.@12_3&12000.mat","160.mat"),    
            ("I.014.DE_0&12000.mat","169.mat"),       ("I.014.DE_1&12000.mat","170.mat"),       ("I.014.DE_2&12000.mat","171.mat"),       ("I.014.DE_3&12000.mat","172.mat"),    
            ("B.014.DE_0&12000.mat","185.mat"),       ("B.014.DE_1&12000.mat","186.mat"),       ("B.014.DE_2&12000.mat","187.mat"),       ("B.014.DE_3&12000.mat","188.mat"),    
            ("O.014.DE.@6_0&12000.mat","197.mat"),    ("O.014.DE.@6_1&12000.mat","198.mat"),    ("O.014.DE.@6_2&12000.mat","199.mat"),    ("O.014.DE.@6_3&12000.mat","200.mat"),    
            ("I.021.DE_0&12000.mat","209.mat"),       ("I.021.DE_1&12000.mat","210.mat"),       ("I.021.DE_2&12000.mat","211.mat"),       ("I.021.DE_3&12000.mat","212.mat"),    
            ("B.021.DE_0&12000.mat","222.mat"),       ("B.021.DE_1&12000.mat","223.mat"),       ("B.021.DE_2&12000.mat","224.mat"),       ("B.021.DE_3&12000.mat","225.mat"),    
            ("O.021.DE.@6_0&12000.mat","234.mat"),    ("O.021.DE.@6_1&12000.mat","235.mat"),    ("O.021.DE.@6_2&12000.mat","236.mat"),    ("O.021.DE.@6_3&12000.mat","237.mat"),    
            ("O.021.DE.@3_0&12000.mat","246.mat"),    ("O.021.DE.@3_1&12000.mat","247.mat"),    ("O.021.DE.@3_2&12000.mat","248.mat"),    ("O.021.DE.@3_3&12000.mat","249.mat"),    
            ("O.021.DE.@12_0&12000.mat","258.mat"),   ("O.021.DE.@12_1&12000.mat","259.mat"),   ("O.021.DE.@12_2&12000.mat","260.mat"),   ("O.021.DE.@12_3&12000.mat","261.mat"),    
            ("I.028.DE_0&12000.mat","3001.mat"),      ("I.028.DE_1&12000.mat","3002.mat"),      ("I.028.DE_2&12000.mat","3003.mat"),      ("I.028.DE_3&12000.mat","3004.mat"),    
            ("B.028.DE_0&12000.mat","3005.mat"),      ("B.028.DE_1&12000.mat","3006.mat"),      ("B.028.DE_2&12000.mat","3007.mat"),      ("B.028.DE_3&12000.mat","3008.mat"),
            ("I.007.FE_0&12000.mat","278.mat"),       ("I.007.FE_1&12000.mat","279.mat"),       ("I.007.FE_2&12000.mat","280.mat"),       ("I.007.FE_3&12000.mat","281.mat"),    
            ("B.007.FE_0&12000.mat","282.mat"),       ("B.007.FE_1&12000.mat","283.mat"),       ("B.007.FE_2&12000.mat","284.mat"),       ("B.007.FE_3&12000.mat","285.mat"),    
            ("O.007.FE.@6_0&12000.mat","294.mat"),    ("O.007.FE.@6_1&12000.mat","295.mat"),    ("O.007.FE.@6_2&12000.mat","296.mat"),    ("O.007.FE.@6_3&12000.mat","297.mat"),    
            ("O.007.FE.@3_0&12000.mat","298.mat"),    ("O.007.FE.@3_1&12000.mat","299.mat"),    ("O.007.FE.@3_2&12000.mat","300.mat"),    ("O.007.FE.@3_3&12000.mat","301.mat"),    
            ("O.007.FE.@12_0&12000.mat","302.mat"),   ("O.007.FE.@12_1&12000.mat","305.mat"),   ("O.007.FE.@12_2&12000.mat","306.mat"),   ("O.007.FE.@12_3&12000.mat","307.mat"),    
            ("I.014.FE_0&12000.mat","274.mat"),       ("I.014.FE_1&12000.mat","275.mat"),       ("I.014.FE_2&12000.mat","276.mat"),       ("I.014.FE_3&12000.mat","277.mat"),    
            ("B.014.FE_0&12000.mat","286.mat"),       ("B.014.FE_1&12000.mat","287.mat"),       ("B.014.FE_2&12000.mat","288.mat"),       ("B.014.FE_3&12000.mat","289.mat"),    
            ("O.014.FE.@3_0&12000.mat","310.mat"),    ("O.014.FE.@3_1&12000.mat","309.mat"),    ("O.014.FE.@3_2&12000.mat","311.mat"),    ("O.014.FE.@3_3&12000.mat","312.mat"),    
            ("O.014.FE.@6_0&12000.mat","313.mat"),    ("I.021.FE_0&12000.mat","270.mat"),       ("I.021.FE_1&12000.mat","271.mat"),       ("I.021.FE_2&12000.mat","272.mat"),       ("I.021.FE_3&12000.mat","273.mat"),    
            ("B.021.FE_0&12000.mat","290.mat"),       ("B.021.FE_1&12000.mat","291.mat"),       ("B.021.FE_2&12000.mat","292.mat"),       ("B.021.FE_3&12000.mat","293.mat"),    
            ("O.021.FE.@6_0&12000.mat","315.mat"),    ("O.021.FE.@3_1&12000.mat","316.mat"),    ("O.021.FE.@3_2&12000.mat","317.mat"),    ("O.021.FE.@3_3&12000.mat","318.mat"),    
            ]


    def __init__(self, debug=False):
        super().__init__(rawfilesdir = "data/raw/cwru",
                         spectdir="data/processed/cwru_spectrograms",
                         sample_rate=12000,
                         url = "https://engineering.case.edu/sites/default/files/",
                         debug=debug)
    
    def load_signal(self, acquisition_maxsize=None, regex_filter=r'B\.007\.DE_0&12000\.mat$'):
        
        regex = re.compile(regex_filter)

        signal = []
        label = []
        for root, dirs, files in os.walk(self.rawfilesdir):
            for file in files:
                filepath = os.path.join(root, file)
               
                if not regex.search(file):
                    continue

                # Load the .mat file into a dictionary
                matlab_file = scipy.io.loadmat(filepath)
                
                # Find keys that match the naming convention of time-domain data (e.g., 'X123_DE_time')
                keys = re.findall(r'X\d{3}_[A-Z]{2}_time', str(matlab_file.keys()))
                
                # Define the accelerometer positions of interest ('DE' for Drive End)
                positions = [ 'DE' ]  # Option to add more positions such as 'FE' for Fan End
                
                # Extract the bearing position from the filename based on its characters
                filename = os.path.basename(filepath)
                bearing_position = positions if filename[6:8] == 'NN' else [filename[6:8]]
                
                # Loop through the matching keys and return the data corresponding to the bearing position
                for key in keys:
                    if key[-7:-5] in bearing_position:
                        # Return the data limited by acquisition_maxsize or return the full data if no limit
                        if acquisition_maxsize:
                            signal.append((np.squeeze(matlab_file[key][:acquisition_maxsize])))
                        else:
                            signal.append(np.squeeze(matlab_file[key]))
                        label.append(filename.split('.')[0])


        self._data = np.array(signal)
        self._label = np.array(label)


    def __str__(self):
        return "CWRU"

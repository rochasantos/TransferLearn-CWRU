from datasets.cwru import CWRU
from datasets.uored import UORED
from datasets.hust import Hust
from datasets.paderborn import Paderborn

from scripts.create_spectrograms import create_spectrograms
from scripts.experiments.kfold import kfold

# PARAMS to create the spectrograms
dataset_name = 'CWRU' # Paderborn, Hust, UORED
window_size = 12000
nperseg = 1024
noverlap = 0
num_segments = 2
target_sr = 12000



if __name__ == '__main__':
    
    
    create_spectrograms(dataset_name, target_sr, window_size, nperseg, 
                        noverlap, num_segments, bearing_type='6203')

    
    
    # kfold()
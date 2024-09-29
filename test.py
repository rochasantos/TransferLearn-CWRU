from datasets.cwru import CWRU
from datasets.paderborn import Paderborn
from datasets.hust import Hust
from datasets.uored import UORED


if __name__ == '__main__':

    """
    Download and Extract files
    """
    # CWRU().download()    
    # Paderborn().download()    
    # Paderborn().extract_rar(remove_rarfile=True)
    # Hust().download()    
    # UORED().download()    

    """
    Create Spectrograms
    """
    from src.feature_engineering.create_cwru_spectrogram import generate_spectrogram
    generate_spectrogram(
        input_dir='data/raw/cwru', 
        output_dir='data/processed/cwru_spectrograms',
        sample_rate=12000)
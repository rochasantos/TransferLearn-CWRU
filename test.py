from datasets.cwru import CWRU
from datasets.paderborn import Paderborn
from datasets.hust import Hust
from datasets.uored import UORED

def test_download():
    """
    Download and Extract files
    """
    # CWRU().download()    
    # Paderborn().download()    
    # Paderborn().extract_rar(remove_rarfile=True)
    # Hust().download()    
    UORED().download()    

def test_create_spectrograms():
    """
    Create Spectrograms
    """
    from src.feature_engineering.create_cwru_spectrogram import generate_cwru_spectrogram
    generate_cwru_spectrogram(
        input_dir='data/raw/cwru', 
        output_dir='data/processed/cwru_spectrograms',
        sample_rate=12000)

    # generate_spectrogram(
    #     input_dir='data/raw/uored',
    #     output_dir='data/processed/uored_spectrograms',
    #     sample_rate=42000)

    # from src.feature_engineering.create_hust_spectrogram import generate_spectrogram
    # generate_spectrogram(
    #     input_dir='data/raw/hust',
    #     output_dir='data/processed/hust_spectrograms',
    #     sample_rate=51200)
    
def main():
    test_create_spectrograms()
    # test_download()

if __name__ == '__main__':
    main()

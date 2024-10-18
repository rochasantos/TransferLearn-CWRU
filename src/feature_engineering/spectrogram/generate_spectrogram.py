import os
from .base import SpectrogramCreator

def generate_spectrogram(data_raw, label, basename, window_size, num_segments, 
                         orig_sr, spectrogram_creator):
      
    # Verificar se o spectrogram_creator é uma instância de SpectrogramCreator
    if not isinstance(spectrogram_creator, SpectrogramCreator):
        raise TypeError("spectrogram_creator must be an instance of SpectrogramCreator")
    
    output_dir = "data/processed/spectrograms"

    # Compute and save spectrograms for 1-second segments of the signal
    n_segments = data_raw.shape[0] // window_size
    n_max_segments = min([num_segments or n_segments, n_segments])
    
    for i in range(0, window_size * n_max_segments, window_size):
        # Save the spectrogram image to the specified output directory
        output_file = os.path.join(output_dir, label, basename+'_{}.png'.format(int((i+1)/window_size)))
        
        if os.path.exists(output_file):
            continue
        
        segment = data_raw[i:i + window_size]
        
        # Generate and save the spectrogram
        spectrogram_creator.generate_spectrogram(segment, output_file, orig_sr)
        
    print(f'{basename}.png files processed.')

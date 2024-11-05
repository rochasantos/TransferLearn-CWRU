import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_spectrogram(data, output_basename, window_size, spec_params, num_segments=None):
    
    # Compute and save spectrograms for 1-second segments of the signal
    n_segments = data.shape[0] // window_size
    n_max_segments = min([num_segments or n_segments, n_segments])
    
    for i in range(0, window_size * n_max_segments, window_size):
        output = os.path.join(output_basename+'#{}.png'.format(int((i+1)/window_size)))
        if os.path.exists(output):
            continue        
        segment = data[i:i + window_size]
        # Calculate the spectrogram
        f, t, Sxx = signal.stft(segment, **spec_params)
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(np.fliplr(abs(Sxx).T).T, cmap='viridis', aspect='auto',
                extent=[t.min(), t.max(), f.min(), f.max()])
        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Number of Samples')
        plt.axis('off')

        # Save the spectrogram
        plt.savefig(output, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f'The processed file was saved in {output}.')

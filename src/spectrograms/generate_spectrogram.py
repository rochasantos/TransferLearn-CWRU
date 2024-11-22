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
        detrended_data = signal.detrend(segment)
        sample = detrended_data[:10500]
        
        # Compute spectrogram
        f, t, Sxx = signal.stft(sample, **spec_params)

        # Plot spectrogram
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(
            np.log(np.abs(Sxx).T + 1e-10),  # Avoid log(0)
            cmap='jet',
            aspect='auto',
            extent=[t.min(), t.max(), f.min(), f.max()],
        )
        plt.ylabel('Frequency [kHz]')
        plt.xlabel('Time [s]')
        plt.axis('off')
        plt.gca().invert_yaxis()

        # Save spectrogram
        plt.savefig(output, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f'The processed file was saved in {output}.')

import os
import numpy as np
from scipy import signal
from datasets import CWRU, Paderborn, Hust, UORED

import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2
from scipy.signal import spectrogram
import matplotlib.animation as animation

def transform_signal_to_scalogram(signal, fs, beta=20, gamma=3, num_voices=18):
    """
    Transforma um sinal bruto em um escalograma usando a Wavelet Morse Analítica.

    Parâmetros:
    - signal: array, o sinal bruto (série temporal).
    - fs: int, frequência de amostragem do sinal (Hz).
    - beta: float, parâmetro de decaimento da wavelet Morse.
    - gamma: float, parâmetro de simetria da wavelet Morse.
    - num_voices: int, número de escalas por oitava.

    Retorna:
    - scalogram: 2D array, o escalograma gerado (tempo x frequência).
    - frequencies: array, frequências associadas ao escalograma.
    """
    # Definir as escalas usando número de vozes por oitava
    max_scale = 256
    scales = np.geomspace(1, max_scale, num=int(num_voices * np.log2(max_scale)))

    # Gerar a transformada wavelet contínua usando morse wavelets
    cwt_matrix = cwt(signal, morlet2, scales, w=gamma)

    # Calcular frequências equivalentes para as escalas
    frequencies = fs / (scales * np.sqrt(beta))

    # Gerar o escalograma (magnitude da CWT)
    scalogram = np.abs(cwt_matrix) ** 2

    return scalogram, frequencies

def generate_escalogram(metainfo, fs, signal_length, num_segments=None):
    
    dataset_name = metainfo[0]["dataset_name"]
    dataset = eval(dataset_name+"()")

    for info in metainfo:
        basename = info["filename"]        
        filepath = os.path.join('data/raw/', dataset_name.lower(), basename+'.mat')
        
        data, label = dataset.load_signal_by_path(filepath)
        
        detrended_data = signal.detrend(data)
        
        n_segments = data.shape[0] // signal_length
        n_max_segments = min([num_segments or n_segments, n_segments])
        
        for i in range(0, signal_length * n_max_segments, signal_length):
            segment = detrended_data[i:i+signal_length]

            # Configurar o segmento desejado
            segment_duration = 1  # Duração do segmento em segundos
            segment_samples = int(segment_duration * fs)  # Número de amostras por segmento
            start = 0  # Início do segmento (modifique conforme necessário)
            end = start + segment_samples
            segment = segment[start:end]

            # wavelet
            scalogram, frequencies = transform_signal_to_scalogram(segment, fs)
            scalogram = scalogram / np.max(scalogram)

            # Plotar o espectrograma
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(scalogram, aspect='auto', extent=[0, segment_duration, frequencies[-1], frequencies[0]], cmap='viridis', origin='lower')
            plt.axis('off')
            plt.gca().invert_yaxis()

            # Save the spectrogram
            output = os.path.join('data/spectrograms', dataset_name.lower()+'_wavelet', label, basename+'#{}.png'.format(int((i+1)/signal_length)))
            plt.savefig(output, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
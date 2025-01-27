import numpy as np
from functools import wraps, partial

# Helper to log method calls
def log_method_call(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        print(f"Calling method: {method.__name__}")
        return method(*args, **kwargs)
    return wrapper

# Data Augmentation strategies
class DataAugmentation:

    @staticmethod
    # @log_method_call
    def local_data_reversing(signal, segment_length):
        """
        Reverse local segments of the signal.
        """
        reversed_signal = signal.copy()
        for i in range(0, len(signal), segment_length):
            segment = reversed_signal[i:i + segment_length]
            reversed_signal[i:i + segment_length] = segment[::-1]
        return reversed_signal

    @staticmethod
    # @log_method_call
    def local_random_reversing(signal, segment_length):
        """
        Reverse a randomly selected local segment of the signal.
        """
        start_idx = np.random.randint(0, len(signal) - segment_length)
        segment = signal[start_idx:start_idx + segment_length]
        signal[start_idx:start_idx + segment_length] = segment[::-1]
        return signal

    @staticmethod
    # @log_method_call
    def global_data_reversing(signal):
        """
        Reverse the entire signal.
        """
        return signal[::-1]

    @staticmethod
    # @log_method_call
    def local_data_zooming(signal, zoom_range=(0.4, 1.6), segment_length=200):
        """
        Scale amplitude of local segments and insert them back.
        """
        zoomed_signal = signal.copy()
        for i in range(0, len(signal), segment_length):
            segment = zoomed_signal[i:i + segment_length]
            zoom_factor = np.random.uniform(*zoom_range)
            zoomed_signal[i:i + segment_length] = segment * zoom_factor
        return zoomed_signal

    @staticmethod
    # @log_method_call
    def global_data_zooming(signal, zoom_range=(0.4, 1.6)):
        """
        Scale the entire signal globally by a random factor.
        """
        zoom_factor = np.random.uniform(*zoom_range)
        return signal * zoom_factor

    @staticmethod
    # @log_method_call
    def local_segment_splicing(signal, segment_length):
        """
        Shuffle local segments of the signal.
        """
        segments = [signal[i:i + segment_length] for i in range(0, len(signal), segment_length)]
        np.random.shuffle(segments)
        return np.concatenate(segments)

    @staticmethod
    # @log_method_call
    def noise_addition(signal, snr_db=20):
        """
        Add Gaussian noise to the signal with a specified SNR.
        """
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    # @staticmethod
    # def get_random_da(signal, segment_length=140):
    #     return [
    #         DataAugmentation.local_data_reversing(signal, segment_length),
    #         DataAugmentation.local_random_reversing(signal, segment_length),
    #         DataAugmentation.global_data_reversing(signal),
    #         DataAugmentation.local_data_zooming(signal, segment_length=segment_length),
    #         DataAugmentation.global_data_zooming(signal),
    #         DataAugmentation.local_segment_splicing(signal, segment_length),
    #         DataAugmentation.noise_addition(signal)
    #     ][random.randint(0, 6)]


augmentations = {
    "local_data_reversing": partial(DataAugmentation.local_data_reversing, segment_length=100),
    "local_random_reversing": partial(DataAugmentation.local_random_reversing, segment_length=100),
    "global_data_reversing": DataAugmentation.global_data_reversing,
    "local_data_zooming": partial(DataAugmentation.local_data_zooming, zoom_range=(0.8, 1.2), segment_length=100),
    "global_data_zooming": partial(DataAugmentation.global_data_zooming, zoom_range=(0.8, 1.2)),
    "local_segment_splicing": partial(DataAugmentation.local_segment_splicing, segment_length=100),
    "noise_addition": partial(DataAugmentation.noise_addition, snr_db=20),
}



# Example usage
if __name__ == "__main__":
    # Simulated signal for demonstration
    simulated_signal = np.sin(np.linspace(0, 2 * np.pi, 1000))  # A sine wave

    da = DataAugmentation()

    # Apply each augmentation
    local_reversed = da.local_data_reversing(simulated_signal, segment_length=100)
    random_reversed = da.local_random_reversing(simulated_signal.copy(), segment_length=100)
    global_reversed = da.global_data_reversing(simulated_signal)
    local_zoomed = da.local_data_zooming(simulated_signal, segment_length=100)
    global_zoomed = da.global_data_zooming(simulated_signal)
    spliced = da.local_segment_splicing(simulated_signal, segment_length=100)
    noisy = da.noise_addition(simulated_signal, snr_db=20)

    # Print the results (for testing purposes)
    print("Local Reversed:", local_reversed[:10])
    print("Random Reversed:", random_reversed[:10])
    print("Global Reversed:", global_reversed[:10])
    print("Local Zoomed:", local_zoomed[:10])
    print("Global Zoomed:", global_zoomed[:10])
    print("Spliced:", spliced[:10])
    print("Noisy:", noisy[:10])

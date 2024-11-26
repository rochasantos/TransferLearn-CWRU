import os
import numpy as np
from PIL import Image

def random_noise(input_dir, output_dir, noise_level=0.05):
    """
    Apply random noise to all images in the input directory and save to the output directory.

    Args:
        input_dir (str): Path to the input directory containing original images.
        output_dir (str): Path to the output directory to save noisy images.
        noise_level (float): Standard deviation of the Gaussian noise to be added (0 to 1 scale).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png') and not file.endswith(('vflip.png', 'hflip.png')):
                filepath = os.path.join(root, file)
                
                # Load image and convert to NumPy array
                img = Image.open(filepath).convert('RGB')
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                
                # Generate random noise
                noise = np.random.normal(0, noise_level, img_array.shape)
                
                # Add noise to image and clip to valid range
                noisy_img_array = np.clip(img_array + noise, 0, 1) * 255.0
                noisy_img = Image.fromarray(noisy_img_array.astype('uint8'))
                
                # Save noisy image
                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_noisy{os.path.splitext(file)[1]}")
                noisy_img.save(output_file)

    print(f"Random noise applied with level {noise_level} and saved successfully.")
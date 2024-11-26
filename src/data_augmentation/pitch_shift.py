import os
from PIL import Image
import numpy as np

def pitch_shift(input_dir, output_dir, shift_steps=20):
    """
    Apply pitch shifting to spectrogram images (.png) by vertically shifting the image.

    Args:
        input_dir (str): Path to the input directory containing spectrogram images.
        output_dir (str): Path to the output directory to save shifted images.
        shift_steps (int): Number of pixels to shift vertically (positive for upward, negative for downward).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png') and not file.endswith(('vflip.png', 'hflip.png', 'shift20.png', 'noisy.png')):
                filepath = os.path.join(root, file)
                
                # Load the image
                img = Image.open(filepath)
                img_array = np.array(img)
                
                # Apply vertical shifting
                shifted_array = np.roll(img_array, shift_steps, axis=0)
                
                # Save the shifted image
                shifted_img = Image.fromarray(shifted_array)
                output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_shift{shift_steps}.png")
                shifted_img.save(output_file)
    
    print(f"Pitch shifting applied with {shift_steps} steps and saved successfully.")

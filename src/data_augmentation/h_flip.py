from PIL import Image
import os

def h_flip(input_dir, output_dir):
    """
    Apply horizontal flip to all images in the input directory and save to the output directory.
    
    Args:
        input_dir (str): Path to the input directory containing original images.
        output_dir (str): Path to the output directory to save horizontally flipped images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png',)) and not file.endswith(('vflip.png', 'hflip.png')):
                filepath = os.path.join(root, file)
                img = Image.open(filepath)
                
                # Apply horizontal flip
                hflip = img.transpose(Image.FLIP_LEFT_RIGHT)
                hflip.save(os.path.join(output_dir, f"{os.path.splitext(file)[0]}_hflip{os.path.splitext(file)[1]}"))
    
    print("Horizontal flip applied and saved successfully.")

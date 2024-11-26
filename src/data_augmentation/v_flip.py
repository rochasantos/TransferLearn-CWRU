from PIL import Image
import os

def v_flip(input_dir, output_dir):
    """
    Apply vertical flip to all images in the input directory and save to the output directory.
    
    Args:
        input_dir (str): Path to the input directory containing original images.
        output_dir (str): Path to the output directory to save vertically flipped images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png',)) and not file.endswith(('vflip.png', 'hflip.png')):
                filepath = os.path.join(root, file)
                img = Image.open(filepath)
                
                # Apply vertical flip
                vflip = img.transpose(Image.FLIP_TOP_BOTTOM)
                vflip.save(os.path.join(output_dir, f"{os.path.splitext(file)[0]}_vflip{os.path.splitext(file)[1]}"))
    
    print("Vertical flip applied and saved successfully.")

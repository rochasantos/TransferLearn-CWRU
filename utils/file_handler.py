import shutil
import os

def copy_file(source_path, destination_path):
    
    try:
        shutil.copy(source_path, destination_path)
        print(f"File successfully copied from {source_path} to {destination_path}.")
    except FileNotFoundError:
        print("Source file not found. Please check the path.")
    except PermissionError:
        print("Permission denied. Check if you have access to the directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


def list_files(directory_path, extension=None):
    
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        
        # Filter files by extension if specified
        if extension:
            files = [file for file in files if file.endswith(extension)]
        
        return files
    except FileNotFoundError:
        print("Directory not found. Please check the path.")
        return []
    except PermissionError:
        print("Permission denied. Check if you have access to the directory.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
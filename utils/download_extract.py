import os
import urllib.request
import shutil
from pyunpack import Archive

from utils.display import display_progress_bar


def download_file(url, dirname, url_suffix, filename):
    """
    Downloads a file from the specified URL and displays a progress bar during the download.
    
    Parameters:
    - url (str): The base URL where the file is located.
    - dirname (str): The directory where the file will be saved.
    - url_suffix (str): The part of the URL that specifies the file to be downloaded.
    - filename (str): The name to save the file as in the specified directory.
    """
    print(f"Downloading the file: {filename}")
    
    try:
        # Request the file size with a HEAD request
        req = urllib.request.Request(url + url_suffix, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        # Define the full path where the file will be saved
        dir_path = os.path.join(dirname, filename)
        
        # Check if the file already exists and if not, download it
        if not os.path.exists(dir_path):
            # Open the connection and the file in write-binary mode
            with urllib.request.urlopen(url + url_suffix) as response, open(dir_path, 'wb') as out_file:
                block_size = 8192  # Define the block size for downloading in chunks
                progress = 0       # Initialize the progress counter
                
                # Download the file in chunks and write each chunk to the file
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress += len(chunk)
                    display_progress_bar(progress, file_size)  # Update the progress bar

            # After the download is complete, display final progress bar with "Download complete"
            display_progress_bar(progress, file_size, done=True)

            # Verify if the downloaded file size matches the expected size
            downloaded_file_size = os.stat(dir_path).st_size
        else:
            downloaded_file_size = os.stat(dir_path).st_size
        
        # If the file size doesn't match, remove the file and try downloading again
        if file_size != downloaded_file_size:
            os.remove(dir_path)
            print("File size incorrect. Downloading again.")
            download_file(url, dirname, url_suffix, filename)
    
    except Exception as e:
        # Handle any errors during the download and retry
        print("Error occurs when downloading file: " + str(e))
        print("Trying to download again")
        download_file(url, dirname, url_suffix, filename)


def extract_rar(dirname, dir_rar, bearing):
    """
    Extracts files from a .rar archive to a specified directory using the pyunpack library.

    Parameters:
    - dirname (str): Base directory where the files will be extracted.
    - dir_rar (str): Subdirectory where the .rar archive is located.
    - bearing (str): Name of the bearing archive to be extracted (without extension).

    Example:
    extract_rar('./data', 'raw', 'bearing_01')
    """
    print("\nExtracting Bearing Data:", bearing)
    dir_bearing_rar = os.path.join(dirname, dir_rar, bearing + ".rar")
    dir_bearing_data = os.path.join(dirname, bearing)

    # Check if the directory already exists, and if so, calculate the number of files
    if not os.path.exists(dir_bearing_data):
        os.makedirs(dir_bearing_data)
        try:
            # Extract the .rar file to the target directory
            Archive(dir_bearing_rar).extractall(dir_bearing_data)

            # Count the number of extracted files
            extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                                       if os.path.isfile(os.path.join(dir_bearing_data, name))])
            
            # Print success message
            print(f"{extracted_files_qnt} files successfully extracted to {dir_bearing_data}")
        except Exception as e:
            print(f"Error extracting {bearing}: {str(e)}")
    else:
        # If directory already exists, count the number of files
        extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                                   if os.path.isfile(os.path.join(dir_bearing_data, name))])
    
    # Check if the number of files in the .rar matches the number of extracted files
    try:
        archive = Archive(dir_bearing_rar)
        rar_files_qnt = len(archive.getnames())
    except Exception as e:
        print(f"Error reading RAR file: {str(e)}")
        rar_files_qnt = 0

    # If the number of files doesn't match, remove the directory and retry extraction
    if rar_files_qnt != extracted_files_qnt:
        shutil.rmtree(dir_bearing_data)
        print("Extracted Files Incorrect. Extracting Again.")
        extract_rar(dirname, dir_rar, bearing)
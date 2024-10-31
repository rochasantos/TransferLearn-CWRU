import os
import urllib.request
from pyunpack import Archive
from utils.display import display_progress_bar


def download_file(url_base, url_suffix, output_path):
    """
    Downloads a file from the specified URL and displays a progress bar during the download.
    
    Parameters:
    - url (str): The base URL where the file is located.
    - url_suffix (str): The part of the URL that specifies the file to be downloaded.
    - output_path (str): The name to save the file as in the specified directory.
    """
    print(f"Downloading the file: {os.path.basename(output_path)}")
    
    try:
        # Request the file size with a HEAD request
        req = urllib.request.Request(url_base + url_suffix, method='HEAD')
        f = urllib.request.urlopen(req)
        file_size = int(f.headers['Content-Length'])

        # Check if the file already exists and if not, download it
        if not os.path.exists(output_path):
            # Open the connection and the file in write-binary mode
            with urllib.request.urlopen(url_base + url_suffix) as response, open(output_path, 'wb') as out_file:
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
            downloaded_file_size = os.stat(output_path).st_size
        else:
            downloaded_file_size = os.stat(output_path).st_size
        
        # If the file size doesn't match, remove the file and try downloading again
        if file_size != downloaded_file_size:
            os.remove(output_path)
            print("File size incorrect. Downloading again.")
            download_file(url_base, url_suffix, output_path)
    
    except Exception as e:
        print("Error occurs when downloading file: " + str(e))
        print("Trying to download again")
        download_file(url_base, url_suffix, output_path)


def extract_rar(file_path, output_dir):
    """ Extract a .rar archive to the specified output directory.

    Parameters:
    - file_path (str): The path to the .rar file to extract.
    - output_dir (str): The directory where the extracted files will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        Archive(file_path).extractall(output_dir)
        print(f"Extraction successful! Files extracted to: {file_path[:-4]}")
    except Exception as e:
        print(f"An error occurred during extraction: {str(e)}")

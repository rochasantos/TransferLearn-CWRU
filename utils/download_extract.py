import os
import urllib.request
import shutil
import rarfile

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


def extract_rar(dirname, rar_path):
    """
    Extracts files from a .rar archive and displays the progress of extraction.

    Parameters:
    dirname (str): The base directory where the extracted files will be stored.
    rar_path (str): The full path to the .rar archive to be extracted.

    Example:
    extract_rar('./data', './data/raw/bearing_01.rar')
    """
    
    # Print a message indicating which archive is being extracted
    print("\nExtracting Bearing Data:", os.path.basename(rar_path))
    
    # Define the path to the .rar file and the target directory for extraction
    dir_bearing_rar = rar_path  # Full path to the .rar file
    dir_bearing_data = os.path.splitext(rar_path)[0]  # Directory with the same name as the .rar file, without the extension
    
    # If the directory for extracted files does not exist, create it and proceed with extraction
    if not os.path.exists(dir_bearing_data):
        os.makedirs(dir_bearing_data)  # Create directory for the extracted files
        file_name = dir_bearing_rar  # Path to the .rar file
        
        # Open the .rar file using the rarfile library
        rf = rarfile.RarFile(file_name)
        total_files = len(rf.namelist())  # Get the total number of files in the .rar archive
        
        # Extract each file from the archive and display the extraction progress
        with rarfile.RarFile(file_name) as rf:
            for i, member in enumerate(rf.infolist(), 1):
                rf.extract(member, path=dirname)  # Extract the current file to the target directory
                # Update and display the progress of the extraction
                done = int(50 * i / total_files)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {i}/{total_files} files extracted", end='')
        
        # Count the number of files that have been extracted into the target directory
        extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                                   if os.path.isfile(os.path.join(dir_bearing_data, name))])
    else:
        # If the directory already exists, count the files already present
        extracted_files_qnt = len([name for name in os.listdir(dir_bearing_data)
                                   if os.path.isfile(os.path.join(dir_bearing_data, name))])
    
    # Reopen the .rar file to verify the number of files inside the archive
    rf = rarfile.RarFile(dir_bearing_rar)
    rar_files_qnt = len(rf.namelist())  # Get the total number of files in the .rar archive
    
    # Compare the number of files in the archive with the number of files extracted
    if rar_files_qnt != extracted_files_qnt + 1:
        # If the number of files does not match, delete the extracted directory and retry extraction
        shutil.rmtree(dir_bearing_data)  # Remove the incomplete extraction directory
        print("Extracted Files Incorrect. Extracting Again.")
        extract_rar(dirname, rar_path)  # Retry the extraction process


def remove_rar_files(directory):
    """
    rRemoves all .rar files in the specified directory.

    Parameters:
    directory (str): The path to the directory where .rar files will be removed.

    Example:
    remove_rar_files('./data/raw')
    """
    
    # Check if the provided directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    # Iterate over all the files in the directory
    for file_name in os.listdir(directory):
        # Create the full path of the file
        file_path = os.path.join(directory, file_name)
        
        # Check if the file is a .rar file and if it is a file (not a directory)
        if file_name.endswith('.rar') and os.path.isfile(file_path):
            try:
                # Remove the .rar file
                os.remove(file_path)
                print(f"Deleted: {file_name}")
            except Exception as e:
                # Handle any exception that occurs during deletion
                print(f"Error deleting {file_name}: {str(e)}")
    
    print("Finished deleting .rar files.")
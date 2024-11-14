import os
import re

def filter_files_with_regex(directory, regex_pattern):
    
    matching_files = []
    pattern = re.compile(regex_pattern)

    for root, _, files in os.walk(directory):
        for file in files:
            if pattern.search(file):  # Verifica se o nome do arquivo corresponde ao padrão
                matching_files.append(os.path.join(root, file))

    return matching_files


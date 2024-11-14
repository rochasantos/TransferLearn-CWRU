import logging
import sys
from datetime import datetime

def generate_filename(name="result"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  
    return f"results/{timestamp}_{name}.txt"

class LoggerWriter:
    def __init__(self, level, name="result"):
        self.level = level
        # Gera o nome do arquivo com base no parâmetro name
        output_file = generate_filename(name)
        
        # Configura o logger para escrever no arquivo especificado e no console
        logging.basicConfig(
            level=logging.INFO,
            format='',
            handlers=[
                logging.FileHandler(output_file), 
                logging.StreamHandler(sys.stdout)
            ]
        )

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass

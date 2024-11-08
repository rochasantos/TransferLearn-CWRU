import logging
import sys
from datetime import datetime

def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  
    return f"results/{timestamp}_result.txt"

output_file = generate_filename()

logging.basicConfig(
    level=logging.INFO,
    format='',  #'%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_file), 
        logging.StreamHandler(sys.stdout)
    ]
)

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass

import logging
import os
import sys

# Set up logging
log_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs.txt'))

class UnicodeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if stream is None:
            stream = sys.stdout
        if hasattr(stream, 'encoding') and stream.encoding.lower() != 'utf-8':
            self.setStream(open(stream.fileno(), mode='w', encoding='utf-8', buffering=1))

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)

# Add this in your initialization section, before running the bot
logging.info("Logging system initialized.")
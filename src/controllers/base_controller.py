from utils.config import Settings, get_settings
from pathlib import Path
import os

class BaseController():
    def __init__(self):
        self.app_settings = get_settings()
        self.BASE_DIR = Path(__file__).resolve().parent.parent

    def get_file_dir(self):
        self.FILE_DIR = self.BASE_DIR / "assets" / "files"
        if not os.path.exists(self.FILE_DIR):
            os.makedirs(self.FILE_DIR , exist_ok=True)
        return self.FILE_DIR
    
    def get_file_ids(self):
        return os.listdir(self.get_file_dir()) 

        
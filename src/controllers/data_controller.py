from .base_controller import BaseController
from models import ResponseSignal
from fastapi import UploadFile
import re
import uuid

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.MbToByte = 1048576        

    def validate_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.ALLOWED_FILE_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        if file.size > self.app_settings.MAX_FILE_SIZE * self.MbToByte:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value
                
        return True, ResponseSignal.FILE_VALIDATION_SUCCESS.value
    
    def generate_unique_filename(self, filename: str):
        random_uuid = str(uuid.uuid4())
        preprocessed_filename = self._preprocess_filename(filename)
        return random_uuid + "_" + preprocessed_filename
    
    def _preprocess_filename(self, filename: str):
        preprocessed_file_name = re.sub('[^\w.]', '', filename).strip()
        preprocessed_file_name.replace(' ', '_')
        return preprocessed_file_name






from enum import Enum

class ResponseSignal(Enum):
    FILE_UPLOAD_SUCCESS = 'File Uploaded Successfuly'
    FILE_UPLOAD_FAILED = 'File Failed To Upload'
    FILE_VALIDATION_SUCCESS = 'File Validated successfully'
    FILE_TYPE_NOT_SUPPORTED = 'File Type Not Supported'
    FILE_SIZE_EXCEEDED = 'File Size Excceds Limits'
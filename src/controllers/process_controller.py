from .base_controller import BaseController
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum

class ProcessController(BaseController):
    def __init__(self):
        super().__init__()

        self.file_dir = self.get_file_dir()

    def get_file_ext(self, file_id: str):
        return file_id.split('.')[-1].lower()
    
    def get_file_loader(self, file_id: str):
        file_ext = self.get_file_ext(file_id)
        file_path = self.file_dir / file_id

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding='utf-8')
        elif file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        else:
            return None
        
    def get_content(self, file_id: str):
        loader = self.get_file_loader(file_id)
        return loader.load()
        
    def process_file_content(self, file_content: list[Document],
                             file_id: str, chunk_size: int = 400,
                             overlap_size: int = 20):
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len 
        )

        file_content_text = [doc.page_content for doc in file_content]
        file_content_meta = [doc.metadata for doc in file_content]

        chunks = text_splitter.create_documents(file_content_text,
                                                file_content_meta)
        
        return chunks
        

    


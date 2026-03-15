from .base_controller import BaseController
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import ProcessingEnum
from providers.base_provider import EmbeddingProvider
from stores.qdrant_store import QdrantStore
import logging

logger = logging.getLogger('uvicorn.error')


class ProcessController(BaseController):
    def __init__(self, embedding_provider: EmbeddingProvider = None,
                 qdrant_store: QdrantStore = None):
        super().__init__()
        self.file_dir = self.get_file_dir()
        self.embedding_provider = embedding_provider
        self.qdrant_store = qdrant_store

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

    # --- Embedding methods ---

    async def embed_file(self, file_id: str, chunks: list[Document]) -> int:
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings = await self.embedding_provider.embed_batch(texts)

        points = [
            {
                "vector": embedding,
                "payload": {
                    "file_id": file_id,
                    "chunk_index": i,
                    "text": text,
                    "metadata": metadata,
                },
            }
            for i, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas))
        ]

        self.qdrant_store.upsert_points(points)
        return len(points)

    async def embed_file_by_id(
        self, file_id: str,
        chunk_size: int = 400, overlap_size: int = 0
    ) -> int:
        file_content = self.get_content(file_id)
        chunks = self.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )
        if not chunks:
            return 0
        return await self.embed_file(file_id, chunks)

    async def embed_new_files(
        self, chunk_size: int = 400, overlap_size: int = 0
    ) -> tuple[int, int]:
        all_file_ids = self.get_file_ids()
        stored_file_ids = self.qdrant_store.get_stored_file_ids()

        new_file_ids = [fid for fid in all_file_ids if fid not in stored_file_ids]
        total_inserted = 0
        last_error = None
        for file_id in new_file_ids:
            try:
                count = await self.embed_file_by_id(
                    file_id, chunk_size, overlap_size
                )
                total_inserted += count
            except Exception as e:
                logger.error(f"Error embedding file {file_id}: {e}")
                last_error = e

        if new_file_ids and total_inserted == 0 and last_error:
            raise last_error

        return len(new_file_ids), total_inserted

    async def reindex_all(
        self, embedding_model: str,
        chunk_size: int = 400, overlap_size: int = 0
    ) -> tuple[int, int]:
        self.qdrant_store.delete_collection()

        embedding_size = await self.embedding_provider.get_embedding_size()
        self.qdrant_store.create_collection(
            embedding_size=embedding_size,
            embedding_model=embedding_model,
        )

        all_file_ids = self.get_file_ids()
        total_inserted = 0
        last_error = None
        for file_id in all_file_ids:
            try:
                count = await self.embed_file_by_id(
                    file_id, chunk_size, overlap_size
                )
                total_inserted += count
            except Exception as e:
                logger.error(f"Error embedding file {file_id}: {e}")
                last_error = e

        if all_file_ids and total_inserted == 0 and last_error:
            raise last_error

        return len(all_file_ids), total_inserted

    


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
import logging
import uuid

logger = logging.getLogger('uvicorn.error')

COLLECTION_MODEL_KEY = "embedding_model"


class QdrantStore:
    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)

    def create_collection(self, embedding_size: int, embedding_model: str,
                          distance: Distance = Distance.COSINE):
        if self.collection_exists():
            return False
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=distance,
            ),
            metadata={COLLECTION_MODEL_KEY: embedding_model},
        )
        return True

    def delete_collection(self):
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

    def get_collection_model(self) -> str | None:
        if not self.collection_exists():
            return None
        info = self.client.get_collection(self.collection_name)
        metadata = getattr(info, 'metadata', None) or {}
        return metadata.get(COLLECTION_MODEL_KEY)

    def get_stored_file_ids(self) -> set[str]:
        if not self.collection_exists():
            return set()

        file_ids = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["file_id"],
                with_vectors=False,
            )
            for point in results:
                if point.payload and "file_id" in point.payload:
                    file_ids.add(point.payload["file_id"])
            if offset is None:
                break
        return file_ids

    def upsert_points(self, points: list[dict]):
        for i in range(0, len(points), 100):
            points_batch = points[i:i + 100]
            structs = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=p["vector"],
                    payload=p["payload"],
                )
                for p in points_batch
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=structs,
            )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": point.payload.get("text", ""),
                "file_id": point.payload.get("file_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "score": point.score,
                "metadata": point.payload.get("metadata", {}),
            }
            for point in results.points
        ]

    def delete_by_file_id(self, file_id: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_id",
                        match=MatchValue(value=file_id),
                    )
                ]
            ),
        )

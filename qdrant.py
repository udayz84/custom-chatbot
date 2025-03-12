import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from text_embeddings import getEmbeddings
load_dotenv()

class QdrantStore:
    def __init__(self, collection_name: str = "url_content", vector_size: int = 1024):
        self.qdrant_url = os.getenv("QDRANT_HOST")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not all([self.qdrant_url, self.qdrant_api_key]):
            raise ValueError("Missing required environment variables")
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist or recreate if dimensions don't match"""
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            # Check if dimensions match
            collection_info = self.client.get_collection(self.collection_name)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != self.vector_size:
                print(f"Recreating collection due to dimension mismatch (current: {current_dim}, required: {self.vector_size})")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection '{self.collection_name}' with dimension {self.vector_size}")

    def store_text_with_embedding(self, texts: List[str], embeddings: List[List[float]], metadata: Dict[str, Any] = None) -> None:
        """
        Store multiple texts and their pre-computed embeddings in Qdrant
        
        Args:
            texts (List[str]): List of texts to store
            embeddings (List[List[float]]): List of pre-computed embeddings for the texts
            metadata (dict): Additional metadata to store with each text
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"Number of texts ({len(texts)}) does not match number of embeddings ({len(embeddings)})")
        
        points = []
        start_id = self.client.count(self.collection_name).count
        
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            if len(embedding) != self.vector_size:
                raise ValueError(f"Embedding dimension {len(embedding)} does not match collection dimension {self.vector_size}")
            
            point_metadata = {
                "text": text,
                **(metadata or {})
            }
            
            point = models.PointStruct(
                id=start_id + idx,
                vector=embedding,
                payload=point_metadata
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Successfully stored {len(texts)} texts and embeddings in Qdrant")


    def search_similar(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar texts based on embedding similarity
        
        Args:
            query_embedding (List[float]): Embedding vector to search for
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: List of dictionaries containing text and metadata
        """
        if len(query_embedding) != self.vector_size:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} does not match collection dimension {self.vector_size}")
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Extract results
        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                **{k: v for k, v in hit.payload.items() if k != "text"}
            })
            
        return results

if __name__ == "__main__":
    # Test data
    mock_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Python is a versatile programming language",
        "Machine learning is transforming technology"
    ]
    
    # Mock embeddings (1024-dimensional vectors with random values between -1 and 1)
    import numpy as np
    np.random.seed(42)  # For reproducibility
    mock_embeddings = [np.random.uniform(-1, 1, 1024).tolist() for _ in range(len(mock_texts))]
    
    # Initialize QdrantStore with correct vector size
    qdrant = QdrantStore(collection_name="test_collection", vector_size=1024)
    
    try:
        # Store the mock data
        qdrant.store_text_with_embedding(
            texts=mock_texts,
            embeddings=mock_embeddings,
            metadata={"source": "test_data"}
        )
        print("Test successful!")
        
        # Test search functionality
        retrieved_results = qdrant.search_similar(mock_embeddings[0], top_k=2)
        print("\nRetrieved results:")
        for result in retrieved_results:
            print(f"Text: {result['text']}")
            print(f"Score: {result['score']}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

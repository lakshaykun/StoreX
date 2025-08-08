"""
ML workflow integration example with sentence transformers
"""

import numpy as np
import storex
from typing import List, Optional, Tuple

class MLVectorDB:
    """Wrapper class for ML workflow integration"""
    
    def __init__(self, storage_path: Optional[str] = None, similarity_metric: str = "cosine"):
        """
        Initialize ML Vector Database
        
        Args:
            storage_path: Path for persistent storage (optional)
            similarity_metric: "cosine", "dot_product", or "euclidean"
        """
        self.store = storex.VectorStore(storage_path or "", auto_save=bool(storage_path))
        
        # Setup similarity metric
        if similarity_metric == "cosine":
            self.metric = storex.CosineSimilarity()
        elif similarity_metric == "dot_product":
            self.metric = storex.DotProductSimilarity()
        elif similarity_metric == "euclidean":
            self.metric = storex.EuclideanSimilarity()
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        self.search_engine = storex.FlatSearchEngine(self.store, self.metric)
        self._doc_id_counter = 0
    
    def add_embeddings(self, 
                      embeddings: np.ndarray, 
                      metadata: Optional[List[dict]] = None,
                      texts: Optional[List[str]] = None) -> List[int]:
        """
        Add embeddings to the database
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            metadata: Optional list of metadata dicts
            texts: Optional list of source texts
            
        Returns:
            List of document IDs
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array")
        
        n_samples = embeddings.shape[0]
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = [{"id": self._doc_id_counter + i} for i in range(n_samples)]
        
        if texts is not None:
            for i, text in enumerate(texts):
                metadata[i]["text"] = text
        
        doc_ids = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            doc_id = self._doc_id_counter
            meta["doc_id"] = doc_id
            
            # Convert numpy array to list
            embedding_list = emb.astype(np.float32).tolist()
            
            # Insert into store
            self.store.insert(embedding_list, meta)
            doc_ids.append(doc_id)
            self._doc_id_counter += 1
        
        return doc_ids
    
    def similarity_search(self, 
                         query_embedding: np.ndarray, 
                         k: int = 5) -> List[Tuple[float, dict]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query vector as numpy array
            k: Number of results to return
            
        Returns:
            List of (similarity_score, metadata) tuples
        """
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1D array")
        
        query_list = query_embedding.astype(np.float32).tolist()
        
        try:
            results = self.search_engine.search(query_list, k)
            return [(score, dict(doc.metadata)) for score, doc in results]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        return {
            "num_vectors": len(self.store),
            "has_storage": self.store.has_storage(),
            "similarity_metric": type(self.metric).__name__
        }
    
    def save(self) -> bool:
        """Save database to disk"""
        return self.store.save()
    
    def load(self) -> bool:
        """Load database from disk"""
        return self.store.load()

def example_with_sentence_transformers():
    """Example using sentence transformers (requires: pip install sentence-transformers)"""
    try:
        from sentence_transformers import SentenceTransformer
        
        print("=== Sentence Transformers Integration Example ===")
        
        # Load a sentence transformer model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample documents
        documents = [
            "The cat sits on the mat",
            "Dogs are loyal pets and great companions", 
            "Machine learning is revolutionizing technology",
            "Python is a versatile programming language",
            "Natural language processing enables AI to understand text",
            "Vector databases store high-dimensional embeddings",
            "Artificial intelligence is transforming industries"
        ]
        
        print(f"Encoding {len(documents)} documents...")
        embeddings = model.encode(documents)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Create ML vector database
        db = MLVectorDB("./ml_vectors.db", similarity_metric="cosine")
        
        # Add embeddings with text metadata
        doc_ids = db.add_embeddings(embeddings, texts=documents)
        print(f"Added {len(doc_ids)} documents to database")
        
        # Perform similarity search
        queries = [
            "pets and animals",
            "programming and coding", 
            "artificial intelligence"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            query_embedding = model.encode([query])[0]
            results = db.similarity_search(query_embedding, k=3)
            
            print("Results:")
            for i, (score, metadata) in enumerate(results):
                text = metadata.get('text', 'N/A')
                print(f"  {i+1}. Score: {score:.4f}")
                print(f"     Text: {text}")
        
        # Show database stats
        stats = db.get_stats()
        print(f"\nDatabase stats: {stats}")
        
        # Save database
        if db.save():
            print("Database saved successfully")
        
    except ImportError:
        print("Sentence transformers not installed. Install with: pip install sentence-transformers")
    except Exception as e:
        print(f"Example failed: {e}")

def example_with_random_embeddings():
    """Example with random embeddings (no external dependencies)"""
    print("=== Random Embeddings Example ===")
    
    # Create database
    db = MLVectorDB("./random_vectors.db", similarity_metric="cosine")
    
    # Generate random embeddings
    np.random.seed(42)
    n_docs = 1000
    embedding_dim = 384  # Common dimension for sentence transformers
    
    embeddings = np.random.random((n_docs, embedding_dim)).astype(np.float32)
    
    # Create metadata
    metadata = [
        {
            "category": f"cat_{i % 10}",
            "importance": float(np.random.random()),
            "source": f"doc_{i}"
        }
        for i in range(n_docs)
    ]
    
    print(f"Adding {n_docs} random {embedding_dim}-dimensional embeddings...")
    doc_ids = db.add_embeddings(embeddings, metadata=metadata)
    
    # Perform searches
    query_embedding = np.random.random(embedding_dim).astype(np.float32)
    results = db.similarity_search(query_embedding, k=5)
    
    print(f"\nSearch results:")
    for i, (score, meta) in enumerate(results):
        print(f"  {i+1}. Score: {score:.4f}, Category: {meta.get('category')}, Source: {meta.get('source')}")
    
    # Stats
    stats = db.get_stats()
    print(f"\nFinal stats: {stats}")
    
    if db.save():
        print("Database saved to disk")

def main():
    print("StoreX ML Integration Examples")
    print("=" * 50)
    
    # Run random embeddings example (always works)
    example_with_random_embeddings()
    
    print("\n" + "=" * 50)
    
    # Try sentence transformers example
    example_with_sentence_transformers()

if __name__ == "__main__":
    main()

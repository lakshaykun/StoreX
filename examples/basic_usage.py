"""
Basic usage example of StoreX Python bindings
"""

import numpy as np
import storex

def main():
    print("=== StoreX Python Binding Example ===")
    
    # Create vector store with persistence
    store = storex.VectorStore("./example_vectors.db", auto_save=True)
    print(f"Created vector store: {store}")
    
    # Create some sample vectors (simulating embeddings from ML models)
    np.random.seed(42)
    embedding_dim = 128
    num_vectors = 100
    
    print(f"\nGenerating {num_vectors} random {embedding_dim}-dimensional vectors...")
    
    # Insert vectors with metadata
    for i in range(num_vectors):
        # Generate random embedding
        embedding = np.random.random(embedding_dim).astype(np.float32).tolist()
        
        # Create metadata
        metadata = {
            "id": i,
            "category": f"category_{i % 5}",
            "source": f"document_{i}",
            "score": float(np.random.random())
        }
        
        # Insert using the convenience method
        store.insert(embedding, metadata)
        
        if i % 20 == 0:
            print(f"  Inserted {i + 1} vectors...")
    
    print(f"\nTotal vectors in store: {len(store)}")
    
    # Test similarity metrics
    print("\n=== Testing Similarity Metrics ===")
    vec1 = np.random.random(10).astype(np.float32).tolist()
    vec2 = np.random.random(10).astype(np.float32).tolist()
    
    cos_sim = storex.cosine_similarity(vec1, vec2)
    dot_sim = storex.dot_product_similarity(vec1, vec2)
    euc_sim = storex.euclidean_similarity(vec1, vec2)
    
    print(f"Cosine similarity: {cos_sim:.4f}")
    print(f"Dot product similarity: {dot_sim:.4f}")
    print(f"Euclidean similarity: {euc_sim:.4f}")
    
    # Test search functionality
    print("\n=== Testing Search ===")
    
    # Create similarity metric and search engine
    metric = storex.CosineSimilarity()
    search_engine = storex.FlatSearchEngine(store, metric)
    
    # Create a query vector
    query = np.random.random(embedding_dim).astype(np.float32).tolist()
    
    try:
        # Search for similar vectors
        results = search_engine.search(query, k=5)
        print(f"Found {len(results)} similar vectors:")
        
        for i, (score, doc) in enumerate(results):
            print(f"  {i+1}. Score: {score:.4f}, Embedding dim: {len(doc.embedding)}")
            print(f"      Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Search failed (this is expected if json filter conversion isn't implemented): {e}")
    
    # Save the store
    print(f"\n=== Saving Store ===")
    if store.save():
        print("Successfully saved vector store to disk")
    else:
        print("Failed to save vector store")
    
    # Test loading
    print("\n=== Testing Load ===")
    new_store = storex.VectorStore("./example_vectors.db")
    if new_store.load():
        print(f"Successfully loaded vector store with {len(new_store)} vectors")
    else:
        print("Failed to load vector store")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()

# StoreX Python Bindings

## Installation

### Prerequisites
```bash
# Install PyBind11 and build tools
pip install pybind11 build wheel setuptools numpy
```

### Build from Source
```bash
# Clone the repository
git clone https://github.com/lakshaykun/StoreX.git
cd StoreX

# Install in development mode
pip install -e .

# Or build wheel for distribution
python -m build
pip install dist/storex-*.whl
```

## Quick Start

### Basic Usage
```python
import numpy as np
import storex

# Create vector store
store = storex.VectorStore("./my_vectors.db", auto_save=True)

# Insert vectors with metadata
embedding = np.random.random(128).astype(np.float32)
metadata = {"id": 1, "category": "example", "text": "sample document"}
store.insert(embedding.tolist(), metadata)

print(f"Store now contains {len(store)} vectors")

# Save to disk
store.save()
```

### ML Workflow Integration
```python
import numpy as np
import storex

# Create ML-friendly wrapper
class MLVectorDB:
    def __init__(self, path=None):
        self.store = storex.VectorStore(path or "", auto_save=bool(path))
        self.metric = storex.CosineSimilarity()
        self.search_engine = storex.FlatSearchEngine(self.store, self.metric)
    
    def add_embeddings(self, embeddings, metadata=None):
        for i, emb in enumerate(embeddings):
            meta = metadata[i] if metadata else {"id": i}
            self.store.insert(emb.tolist(), meta)
    
    def search(self, query, k=5):
        try:
            return self.search_engine.search(query.tolist(), k)
        except:
            return []  # Handle search errors gracefully

# Usage
db = MLVectorDB("./ml_vectors.db")

# Add embeddings from your ML model
embeddings = np.random.random((100, 384))  # Sentence transformer dimension
metadata = [{"text": f"document_{i}"} for i in range(100)]
db.add_embeddings(embeddings, metadata)

# Search
query = np.random.random(384)
results = db.search(query, k=5)
```

### With Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
import storex

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create database
db = storex.VectorStore("./embeddings.db")

# Process documents
documents = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(documents)

# Store embeddings
for emb, text in zip(embeddings, documents):
    db.insert(emb.tolist(), {"text": text})

# Search
query = "search query"
query_emb = model.encode([query])[0]

# Use similarity metrics
similarity = storex.cosine_similarity(query_emb.tolist(), embeddings[0].tolist())
print(f"Similarity: {similarity}")
```

## API Reference

### VectorStore
- `VectorStore()` - Create without persistence
- `VectorStore(path, auto_save=True)` - Create with storage
- `insert(document)` - Insert Document object
- `insert(embedding, metadata)` - Insert with lists/dicts
- `get_all()` - Get all documents
- `save()` - Save to disk
- `load()` - Load from disk
- `size()` / `len()` - Get count
- `clear()` - Remove all documents

### Document
- `Document(embedding, metadata)` - Create document
- `.embedding` - Vector data
- `.metadata` - Metadata dictionary

### Similarity Metrics
- `CosineSimilarity()` - Cosine similarity
- `DotProductSimilarity()` - Dot product
- `EuclideanSimilarity()` - Euclidean distance

### Search Engines
- `FlatSearchEngine(store, metric)` - Brute force search
- `LSHSearchEngine(store, metric, num_hashes, num_tables)` - LSH search
- `.search(query, k, filter)` - Search for similar vectors

### Utility Functions
- `cosine_similarity(vec1, vec2)` - Direct cosine similarity
- `dot_product_similarity(vec1, vec2)` - Direct dot product
- `euclidean_similarity(vec1, vec2)` - Direct euclidean similarity

## Testing

```bash
# Install test dependencies
pip install pytest numpy

# Run tests
python -m pytest python/tests/ -v
```

## Examples

See the `examples/` directory for complete examples:
- `basic_usage.py` - Basic vector operations
- `ml_integration.py` - ML workflow integration

## Troubleshooting

### Build Issues
- Ensure you have a C++17 compatible compiler
- Install PyBind11: `pip install pybind11`
- Check that all source files exist in setup.py

### Import Errors
- The module will be called `storex` after building
- Import errors before building are expected

### Performance Tips
- Use `auto_save=False` for bulk operations
- Call `save()` manually when needed
- Use appropriate similarity metrics for your use case
- Consider LSH search for large datasets

## Integration with Popular ML Libraries

### LangChain
```python
from langchain.vectorstores.base import VectorStore as LangChainVectorStore

class StoreXVectorStore(LangChainVectorStore):
    def __init__(self, storage_path=None):
        self.db = storex.VectorStore(storage_path or "")
    
    def add_texts(self, texts, metadatas=None):
        # Implement text to embedding conversion
        pass
```

### Hugging Face
```python
from transformers import AutoTokenizer, AutoModel
import storex

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Use with StoreX...
```

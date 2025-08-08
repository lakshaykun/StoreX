"""StoreX Python Bindings

High-performance vector database for ML workflows.
"""

try:
    from .storex import (
        VectorStore,
        Document,
        CosineSimilarity,
        DotProductSimilarity, 
        EuclideanSimilarity,
        FlatSearchEngine,
        LSHSearchEngine,
        cosine_similarity,
        dot_product_similarity,
        euclidean_similarity,
        __version__
    )
except ImportError as e:
    raise ImportError(
        "StoreX C++ extension not found. Please install with: pip install storex"
    ) from e

__all__ = [
    'VectorStore',
    'Document', 
    'CosineSimilarity',
    'DotProductSimilarity',
    'EuclideanSimilarity',
    'FlatSearchEngine',  
    'LSHSearchEngine',
    'cosine_similarity',
    'dot_product_similarity',
    'euclidean_similarity',
    '__version__'
]

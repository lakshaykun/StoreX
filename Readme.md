## what am i making? 
    vector databse

## tell me how should i approch it, how should i think about this project and where to start what to learn, what to use, which things to consider while writing the code.
Ask: “What is this system supposed to do really well?”
    The answer for a vector database is:
        Fast similarity search over high-dimensional vectors
        Scalable to millions+ entries
        Filterable with metadata
        Persistent (saves & loads data)
        Bindable (usable via Python, LangChain, API)
        So always optimize for performance, extensibility, and correctness.

## Project Approach – Step-by-Step Plan
    Phase 1: Plan the MVP (Minimum Viable Product)
    Just get the basics working:
        Vector insertion
        Brute-force similarity search (e.g., cosine similarity)
        Metadata storage
        In-memory only

    Phase 2: Upgrade with ANN Indexing
        Methods to include:
            HNSW (Hierarchical Navigable Small World) graphs
            ANNOY (Approximate Nearest Neighbors Oh Yeah)
            LSH (Locality Sensitive Hashing)
        Why? Because brute-force search is too slow for large datasets.
        10x+ faster than brute-force
        Keeps you competitive with tools like FAISS/Pinecone

    Phase 3: Add Persistence + Python Bindings
        Now make your system practical:
            Save/load to/from disk
            Expose API to Python (using pybind11)



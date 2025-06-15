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
        Use or build a real indexing algorithm (HNSW or IVF):
        10x+ faster than brute-force
        Keeps you competitive with tools like FAISS/Pinecone

    Phase 3: Add Persistence + Python Bindings
        Now make your system practical:
            Save/load to/from disk
            Expose API to Python (using pybind11)
            Start writing CLI or REST interface


## What to Learn / Polish
       Area	                                   Why	                                      Resource
C++ STL & Memory Mgmt	           Core of performance & containers	      cppreference.com, or Book: Effective C++
Vector Similarity Metrics	       Cosine, L2, Dot Product	              Paper/blogs on embedding spaces
ANN Algorithms (HNSW)	           Key to performance	                  HNSW Paper, hnswlib
File I/O & Serialization	       For persistence	                      Learn binary I/O, or use protobuf / msgpack
Multithreading in C++	           For fast concurrent searches	          std::thread / OpenMP
Python Bindings	                   To integrate with AI workflows         pybind11
LangChain plugin basics		       To serve it in AI chains               LangChain docs




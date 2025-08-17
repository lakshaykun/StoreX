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
    Phase 1: MVP
        a. Virtual Classes
        b. Flat Index
        c. Document Structure

    Phase 2:
        a. Unit tests
        b. Storage Module

    Phase 3:
        LSH

    Phase 4:
        Annoy
    
    Phase 5:
        HNSW

    Phase 6:
        a. Python Binding 
        b. Complete unit testing
        c. Testing on millions of data
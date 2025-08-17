# StoreX

StoreX is a high-performance C++ vector database system designed for similarity search and document storage. It supports various similarity metrics, indexing strategies (including LSH for approximate nearest neighbor search), and persistent storage with SQLite.

## Features

- **Multiple Similarity Metrics**: Cosine, Euclidean, and Jaccard similarity
- **Flexible Indexing**: FlatIndex for exact search and LSH for approximate search
- **Persistent Storage**: SQLite-based storage for documents and embeddings
- **Metadata Support**: JSON-based metadata for rich document attributes
- **Vector Operations**: High-performance vector similarity computations
- **Comprehensive Testing**: 74 unit tests covering all components

## Directory Structure

```
StoreX/
├── bin/                # Compiled binaries
├── include/            # Header files
├── obj/                # Object files
├── src/                # Source files
├── test/               # Test files
├── Makefile            # Makefile for building the project
└── README.md           # Project documentation
```

## Build Instructions

To build the project, run the following command:

```bash
make
```

This will compile the source files and create the necessary binaries in the `bin/` directory.

## Testing

The project includes comprehensive unit tests that cover all major components:

### Running Tests

To run the tests, use the following command:

```bash
make test
```

Or you can run the test binary directly:

```bash
./bin/tests
```

### Test Coverage

The unit test suite includes tests for:

- **Metadata**: JSON-based metadata handling, parsing, and operations
- **Document**: Document creation, embedding storage, and metadata integration
- **Similarity Metrics**: Cosine, Euclidean, and Jaccard similarity computations
- **Collection**: Document storage, retrieval, and management
- **Storage**: SQLite-based persistence layer
- **Indexing**: FlatIndex and LSH-based vector indexing
- **Vector Store**: High-level API combining all components
- **Integration Tests**: End-to-end workflow testing

### Test Results

The test suite runs 74 comprehensive tests covering:
- Basic functionality of all components
- Error handling and edge cases
- Integration between different modules
- Performance characteristics of similarity computations

## Cleaning Up

To clean up the generated files, you can run:

```bash
make clean
```

This will remove the `bin/`, `obj/`, and any other generated files.
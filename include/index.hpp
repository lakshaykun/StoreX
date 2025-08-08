#pragma once
#include <string>
#include <vector>
#include "document.hpp"
using std::string;
using std::vector;

// virtual class to create indexing for vector store
class Index {
public:
    virtual ~Index() = default;

    // Method to insert a key-value pair into the index
    virtual void insert(Document& doc) = 0;

    // Method to search for a document by metadata
    virtual vector<Document> search(const Metadata& meta) const = 0;

    // Method to search for a document by embedding
    virtual vector<Document> search(const vector<float>& embedding) const = 0;

    // Method to search for a document by metadata and embedding
    virtual vector<Document> search(const Metadata& meta, const vector<float>& embedding) const = 0;
};
#pragma once
#include <string>
#include <vector>
#include "document.hpp"
#include "collection.hpp"
#include "similarity.hpp"

using std::shared_ptr;
using std::make_shared;
using std::vector;

// Abstract virtual class to create indexing for vector store
class Index {
protected:
    shared_ptr<Collection> collection;
    shared_ptr<Similarity> similarity;
public:
    virtual ~Index() = default;

    // Constructor to initialize the index with a collection and similarity measure
    Index(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
        : collection(std::move(coll)), similarity(std::move(sim)) {}

    // Method to insert a document into the index
    virtual int insert(Document& doc) = 0;

    // Method to search for documents of a specific metadata
    virtual vector<Document> search(const Metadata& meta, int k) const = 0;

    // Method to search for top k similar documents by embedding
    virtual vector<Document> search(const vector<float>& embedding, int k) const = 0;

    // Method to search for top k similar documents by embedding with same metadata
    virtual vector<Document> search(const Metadata& meta, const vector<float>& embedding, int k) const = 0;
};
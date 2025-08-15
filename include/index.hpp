#pragma once
#include <string>
#include <vector>
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
        : collection(coll), similarity(sim) {}

    // Method to insert a document into the index
    virtual size_t insert(Document& doc) = 0;

    // Method to update a document in the index
    virtual size_t update(size_t id, Document& doc) = 0;

    // Method to search for documents of a specific metadata
    virtual vector<Document> search(const Metadata& meta, size_t k) const = 0;

    // Method to search for top k similar documents by embedding
    virtual vector<Document> search(const vector<float>& embedding, size_t k) const = 0;

    // Method to search for similar documents and their scores by embedding
    virtual vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k) const = 0;

    // Method to search for top k similar documents by embedding with same metadata
    virtual vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k) const = 0;

    // Method to search for similar documents and their scores by embedding with same metadata
    virtual vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const = 0;
};

class FlatIndex : public Index {
public:
    FlatIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    size_t insert(Document& doc) override;

    size_t update(size_t id, Document& doc) override;

    vector<Document> search(const Metadata& meta, size_t k) const override;

    vector<Document> search(const vector<float>& embedding, size_t k) const override;

    vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k) const override;

    vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k) const override;

    vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const override;

    ~FlatIndex() override = default;
};
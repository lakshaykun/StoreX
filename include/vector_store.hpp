#pragma once
#include "index.hpp"
#include <memory>

using std::shared_ptr;
using std::make_shared;

class vector_store {
private:
    shared_ptr<Collection> collection;
    shared_ptr<Similarity> similarity;
    shared_ptr<Index> index;
public:
    // Default constructor without persistence
    vector_store(size_t vecSize);

    // custom setup constructor
    vector_store(shared_ptr<Index> ind, shared_ptr<Collection> coll);

    // Method to insert a document into the vector store
    size_t insert(Document& doc);

    // Method to insert multiple documents into the vector store
    vector<size_t> insert(vector<Document>& docs);

    // Method to update a document in the vector store
    void update(size_t id, Document& doc);

    // Method to search for k documents of a specific metadata
    vector<Document> search(const Metadata& meta, size_t k);

    // Method to search for top k similar documents by embedding
    vector<Document> search(const vector<float>& embedding, size_t k);

    // Method to search for similar documents and their scores by embedding
    vector<std::pair<float, Document>> searchWithScores(const vector<float>& embedding, size_t k);

    // Method to search for top k similar documents by embedding with same metadata
    vector<Document> search(const Metadata& meta, const vector<float>& embedding, size_t k);

    // Method to search for similar documents and their scores by embedding with same metadata
    vector<std::pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k);

    // Method to fetch id of a document by metadata
    size_t fetchId(const Metadata& meta);

    // Method to fetch id of a document by embedding
    size_t fetchId(const vector<float>& embedding);

    // Method to fetch id of a document
    size_t fetchId(const Document& doc);

    // Method to fetch document by id
    Document fetchDocument(size_t id);

    // Method to get the collection
    shared_ptr<Collection> getCollection();
};
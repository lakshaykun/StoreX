#pragma once
#include "collection.hpp"
#include "document.hpp"
#include "index.hpp"
#include "similarity.hpp"
#include <memory>
using std::shared_ptr;
using std::make_shared;

class vector_store {
private:
    shared_ptr<Collection> collection;
    shared_ptr<Index> index;
    shared_ptr<Similarity> similarity;
public:
    // Default constructor without persistence
    vector_store();

    // custom setup constructor
    vector_store(shared_ptr<Index> ind, shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    // Method to insert a document into the vector store
    void insert(Document& doc);

    // Method to search for a document by metadata
    vector<Document> search(const Metadata& meta) const;

    // Method to search for a document by embedding
    vector<Document> search(const vector<float>& embedding) const;

    // Method to search for a document by metadata and embedding
    vector<Document> search(const Metadata& meta, const vector<float>& embedding) const;
};
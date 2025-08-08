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
    vector_store(shared_ptr<Index> ind, shared_ptr<Collection> coll);

    // Method to insert a document into the vector store
    int insert(Document& doc);

    // Method to search for k documents of a specific metadata
    vector<Document> search(const Metadata& meta, int k);

    // Method to search for top k similar documents by embedding
    vector<Document> search(const vector<float>& embedding, int k);

    // Method to search for top k similar documents by embedding with same metadata
    vector<Document> search(const Metadata& meta, const vector<float>& embedding, int k);

    // Method to fetch id of a document by metadata
    int fetchId(const Metadata& meta);

    // Method to fetch id of a document by embedding
    int fetchId(const vector<float>& embedding);

    // Method to fetch document by id
    Document fetchDocument(const string& id);

    // Method to get the collection
    shared_ptr<Collection> getCollection();
};
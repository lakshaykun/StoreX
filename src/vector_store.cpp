# include "vector_store.hpp"
#include "collection.hpp"
#include "index.hpp"
#include "flat_index.cpp"
#include "similarity.cpp"
#include <vector>
#include <memory>
#include <string>

using std::vector;
using std::string;
using std::move;

// Default constructor without persistence
vector_store::vector_store() 
    : collection(make_shared<Collection>()), 
    similarity(make_shared<CosineSimilarity>()),
    index(make_shared<FlatIndex>(collection, similarity)) {}

// Custom setup constructor
vector_store::vector_store(shared_ptr<Index> ind, shared_ptr<Collection> coll)
    : index(move(ind)), collection(move(coll)) {}

// Method to insert a document into the vector store
int vector_store::insert(Document& doc) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
        return -1; // Indicate failure
    }
    return index->insert(doc);
}

// Method to search for k documents of a specific metadata
vector<Document> vector_store::search(const Metadata& meta, int k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->search(meta, k);
}

// Method to search for top k similar documents by embedding
vector<Document> vector_store::search(const vector<float>& embedding, int k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->search(embedding, k);
}

// Method to search for top k similar documents by embedding with same metadata
vector<Document> vector_store::search(const Metadata& meta, const vector<float>& embedding, int k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->search(meta, embedding, k);
}

// Method to fetch id of a document by metadata
int vector_store::fetchId(const Metadata& meta) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    for (int i=0; i<collection->documents.size(); ++i) {
        if (collection->documents[i].getMetadata() == meta) {
            return i;
        }
    }
    return -1; // Indicate not found
}

// Method to fetch id of a document by embedding
int vector_store::fetchId(const vector<float>& embedding) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    for (int i=0; i<collection->documents.size(); ++i) {
        if (collection->documents[i].getEmbedding() == embedding) {
            return i;
        }
    }
    return -1; // Indicate not found
}

// Method to fetch document by id
Document vector_store::fetchDocument(const string& id) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    int docId = std::stoi(id);
    if (docId < 0 || docId >= collection->documents.size()) {
        throw std::out_of_range("Document ID out of range.");
    }
    return collection->documents[docId];
}

// Method to get the collection
shared_ptr<Collection> vector_store::getCollection() {
    return collection;
}
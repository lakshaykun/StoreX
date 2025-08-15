#pragma once
#include "document.hpp"
#include <vector>
using std::vector;

class Collection {
private:
    vector<Document> documents;
public:

    // default destructor
    ~Collection() = default;
    
    // Method to insert a document into the collection
    size_t insert(Document& doc) {
        documents.emplace_back(doc);
        return documents.size() - 1; // Return the index of the inserted document
    }

    // Method to update a document in the collection
    size_t update(size_t id, Document& doc) {
        if (id >= documents.size()) {
            return 0; // Indicate failure
        }
        documents[id] = doc; // Update the document at the given index
        return id; // Return the index of the updated document
    }

    // Method to get the collection of documents
    const vector<Document>& getDocuments() const {
        return documents;
    }

    // Method to get the size of the collection
    size_t size() const {
        return documents.size();
    }
};
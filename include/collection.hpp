#pragma once
#include "document.hpp"
#include <vector>
using std::vector;

class Collection {
public:
    vector<Document> documents;
    
    ~Collection() = default;
    
    // Method to insert a document into the collection
    int insert(Document& doc) {
        documents.push_back(doc);
        return documents.size() - 1; // Return the index of the inserted document
    }
};
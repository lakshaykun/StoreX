#pragma once
#include "document.hpp"
#include <vector>
using std::vector;

class Collection {
    vector<Document> documents;
public:
    virtual ~Collection() = default;
    
    // Method to insert a document into the collection
    virtual void insert(Document& doc) = 0;
};
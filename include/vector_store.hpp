#pragma once
#include "document.hpp"
#include "storage.hpp"
#include <vector>
#include <memory>
using namespace std;

class VectorStore {
private:
    vector<Document> Documents;
    unique_ptr<Storage> storage;
    bool auto_save;

public:
    // Default constructor - no persistence
    VectorStore();
    
    // Constructor with storage path - enables persistence
    VectorStore(const string& storage_path, bool auto_save = true);
    
    // Insert a single document
    void insert(const Document& v);
    
    // Insert multiple documents
    void insert(const vector<Document>& docs);
    
    // Get all documents
    const vector<Document>& getAll() const;
    
    // Save all documents to storage
    bool save();
    
    // Load documents from storage
    bool load();
    
    // Clear all documents from memory and storage
    bool clear();
    
    // Get the number of documents
    size_t size() const;
    
    // Check if storage is enabled
    bool hasStorage() const;
};

#include "vector_store.hpp"
#include <iostream>

// Default constructor - no persistence
VectorStore::VectorStore() : auto_save(false) {}

// Constructor with storage path - enables persistence
VectorStore::VectorStore(const string& storage_path, bool auto_save) 
    : storage(make_unique<Storage>(storage_path)), auto_save(auto_save) {
    // Load existing documents if storage file exists
    if (storage->file_exists()) {
        load();
    }
}

// Insert a single document
void VectorStore::insert(const Document& v) {
    Documents.push_back(v);
    
    // Auto-save if enabled and storage is available
    if (auto_save && storage) {
        if (!storage->save_document(v)) {
            std::cerr << "Warning: Failed to auto-save document to storage" << std::endl;
        }
    }
}

// Insert multiple documents
void VectorStore::insert(const vector<Document>& docs) {
    Documents.insert(Documents.end(), docs.begin(), docs.end());
    
    // Auto-save if enabled and storage is available
    if (auto_save && storage) {
        if (!save()) {
            std::cerr << "Warning: Failed to auto-save documents to storage" << std::endl;
        }
    }
}

// Get all documents
const vector<Document>& VectorStore::getAll() const {
    return Documents;
}

// Save all documents to storage
bool VectorStore::save() {
    if (!storage) {
        std::cerr << "Error: No storage configured" << std::endl;
        return false;
    }
    
    return storage->save_documents(Documents);
}

// Load documents from storage
bool VectorStore::load() {
    if (!storage) {
        std::cerr << "Error: No storage configured" << std::endl;
        return false;
    }
    
    Documents = storage->load_documents();
    return true;
}

// Clear all documents from memory and storage
bool VectorStore::clear() {
    Documents.clear();
    
    if (storage) {
        return storage->clear_storage();
    }
    
    return true;
}

// Get the number of documents
size_t VectorStore::size() const {
    return Documents.size();
}

// Check if storage is enabled
bool VectorStore::hasStorage() const {
    return storage != nullptr;
}

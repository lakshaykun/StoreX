#pragma once
#include <string>
#include <vector>
#include <fstream>
#include "document.hpp"
#include "json.hpp"

using namespace std;

/**
 * Storage class for persisting and loading vectors and metadata using JSON Lines format
 * Each line in the file contains one JSON object representing a document
 */
class Storage {
private:
    string storage_path;  // Path to the storage file
    
public:
    /**
     * Constructor - sets the storage file path
     * @param path Path to the storage file
     */
    Storage(const string& path);
    
    /**
     * Save a single document to the storage file (append mode)
     * @param document Document containing vector and metadata to save
     * @return true if successful, false otherwise
     */
    bool save_document(const Document& document);
    
    /**
     * Save multiple documents to the storage file (overwrite mode)
     * @param documents Vector of documents to save
     * @return true if successful, false otherwise
     */
    bool save_documents(const vector<Document>& documents);
    
    /**
     * Load all documents from the storage file
     * @return Vector of loaded documents
     */
    vector<Document> load_documents();
    
    /**
     * Clear all data from the storage file
     * @return true if successful, false otherwise
     */
    bool clear_storage();
    
    /**
     * Check if storage file exists
     * @return true if file exists, false otherwise
     */
    bool file_exists();
    
    /**
     * Get the number of documents in storage
     * @return Number of documents stored
     */
    size_t document_count();
};
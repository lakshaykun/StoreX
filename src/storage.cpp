#include "storage.hpp"
#include <iostream>
#include <sstream>

using namespace std;

/**
 * Constructor - Initialize storage with file path
 */
Storage::Storage(const string& path) : storage_path(path) {}

/**
 * Convert MetadataValue (variant) to JSON
 */
json metadata_value_to_json(const MetadataValue& value) {
    json j;
    if (holds_alternative<string>(value)) {
        j = get<string>(value);
    } else if (holds_alternative<int>(value)) {
        j = get<int>(value);
    } else if (holds_alternative<float>(value)) {
        j = get<float>(value);
    }
    return j;
}

/**
 * Convert JSON to MetadataValue (variant)
 */
MetadataValue json_to_metadata_value(const json& j) {
    if (j.is_string()) {
        return j.get<string>();
    } else if (j.is_number_integer()) {
        return j.get<int>();
    } else if (j.is_number_float()) {
        return j.get<float>();
    }
    // Default to empty string if type is unknown
    return string("");
}

/**
 * Convert Metadata to JSON object
 */
json metadata_to_json(const Metadata& metadata) {
    json j = json::object();
    for (const auto& pair : metadata) {
        j[pair.first] = metadata_value_to_json(pair.second);
    }
    return j;
}

/**
 * Convert JSON object to Metadata
 */
Metadata json_to_metadata(const json& j) {
    Metadata metadata;
    if (j.is_object()) {
        for (auto it = j.begin(); it != j.end(); ++it) {
            metadata[it.key()] = json_to_metadata_value(it.value());
        }
    }
    return metadata;
}

/**
 * Convert Document to JSON object
 */
json document_to_json(const Document& document) {
    json j;
    j["embedding"] = document.embedding;  // Store vector as JSON array
    j["metadata"] = metadata_to_json(document.metadata);  // Store metadata as JSON object
    return j;
}

/**
 * Convert JSON object to Document
 */
Document json_to_document(const json& j) {
    Document doc;
    doc.embedding = j["embedding"].get<vector<float>>();
    doc.metadata = json_to_metadata(j["metadata"]);
    return doc;
}

/**
 * Save a single document to storage file (append mode)
 */
bool Storage::save_document(const Document& document) {
    try {
        // Open file in append mode
        ofstream file(storage_path, ios::app);
        if (!file.is_open()) {
            cerr << "Error: Could not open storage file for writing: " << storage_path << endl;
            return false;
        }
        
        // Convert document to JSON and write as single line
        json j = document_to_json(document);
        file << j.dump() << "\n";  // JSON Lines format: one JSON per line
        
        file.close();
        return true;
    } catch (const exception& e) {
        cerr << "Error saving document: " << e.what() << endl;
        return false;
    }
}

/**
 * Save multiple documents to storage file (overwrite mode)
 */
bool Storage::save_documents(const vector<Document>& documents) {
    try {
        // Open file in overwrite mode
        ofstream file(storage_path);
        if (!file.is_open()) {
            cerr << "Error: Could not open storage file for writing: " << storage_path << endl;
            return false;
        }
        
        // Write each document as a separate JSON line
        for (const auto& document : documents) {
            json j = document_to_json(document);
            file << j.dump() << "\n";
        }
        
        file.close();
        return true;
    } catch (const exception& e) {
        cerr << "Error saving documents: " << e.what() << endl;
        return false;
    }
}

/**
 * Load all documents from storage file
 */
vector<Document> Storage::load_documents() {
    vector<Document> documents;
    
    try {
        ifstream file(storage_path);
        if (!file.is_open()) {
            cerr << "Warning: Could not open storage file for reading: " << storage_path << endl;
            return documents;  // Return empty vector
        }
        
        string line;
        // Read each line as a separate JSON object
        while (getline(file, line)) {
            if (line.empty()) continue;  // Skip empty lines
            
            try {
                json j = json::parse(line);
                Document doc = json_to_document(j);
                documents.push_back(doc);
            } catch (const exception& e) {
                cerr << "Error parsing JSON line: " << e.what() << endl;
                // Continue reading other lines
            }
        }
        
        file.close();
    } catch (const exception& e) {
        cerr << "Error loading documents: " << e.what() << endl;
    }
    
    return documents;
}

/**
 * Clear all data from storage file
 */
bool Storage::clear_storage() {
    try {
        // Open file in truncate mode to clear contents
        ofstream file(storage_path, ios::trunc);
        if (!file.is_open()) {
            cerr << "Error: Could not open storage file for clearing: " << storage_path << endl;
            return false;
        }
        file.close();
        return true;
    } catch (const exception& e) {
        cerr << "Error clearing storage: " << e.what() << endl;
        return false;
    }
}

/**
 * Check if storage file exists
 */
bool Storage::file_exists() {
    ifstream file(storage_path);
    return file.good();
}

/**
 * Get the number of documents in storage
 */
size_t Storage::document_count() {
    size_t count = 0;
    
    try {
        ifstream file(storage_path);
        if (!file.is_open()) {
            return 0;
        }
        
        string line;
        // Count non-empty lines
        while (getline(file, line)) {
            if (!line.empty()) {
                count++;
            }
        }
        
        file.close();
    } catch (const exception& e) {
        cerr << "Error counting documents: " << e.what() << endl;
        return 0;
    }
    
    return count;
}
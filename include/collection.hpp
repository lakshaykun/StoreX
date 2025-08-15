#pragma once
#include "document.hpp"
#include "storage.hpp"
#include <vector>
#include <memory>
using std::vector;
using std::shared_ptr;

class Collection {
private:
    vector<Document> documents;
    bool storageEnabled = false; // Flag to indicate if storage is enabled
    shared_ptr<Storage> storage; // Pointer to the storage object if needed
public:
    // default destructor
    ~Collection() = default;

    // storage enabled constructor with default file name
    Collection(bool enableStorage = false) : storageEnabled(enableStorage) {
        if (storageEnabled) {
            storage = std::make_shared<Storage>();
            // Check if storage was initialized successfully
            if (!storage) {
                throw std::runtime_error("Failed to initialize storage");
            } 
            // load the documents from storage if needed
            try {
                documents = storage->load();
            } catch (const std::exception& e) {
                throw std::runtime_error("Error loading documents: " + std::string(e.what()));
            }
        }
    }

    // storage enabled constructor with custom file name
    Collection(const string& filename) : storageEnabled(true) {
        storage = std::make_shared<Storage>(filename);
    }

    // Method to enable or disable storage
    void enableStorage(bool enable, const string& filename = "db/storex.db") {
        storageEnabled = enable;
        if (enable && !storage) {
            storage = std::make_shared<Storage>(filename);
            // Check if storage was initialized successfully
            if (!storage) {
                throw std::runtime_error("Failed to initialize storage");
            }
            // overwrite existing documents in the new storage
            try {
                storage->insert(documents);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error inserting documents: " + std::string(e.what()));
            }
        }
    }

    // Method to insert a document into the collection
    size_t insert(Document& doc) {
        documents.emplace_back(doc);
        size_t id = documents.size() - 1;
        if (storageEnabled) {
            storage->insert(id, doc);
        }
        return id;
    }

    // Method to update a document in the collection
    void update(size_t id, Document& doc) {
        if (id >= documents.size()) {
            throw std::out_of_range("Document ID out of range");
        }
        documents[id] = doc; // Update the document at the given index
        if (storageEnabled) {
            storage->update(id, doc);
        }
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
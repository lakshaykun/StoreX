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
    size_t vecSize = 0; // Size of the vectors in the collection
public:
    // default destructor
    ~Collection() = default;

    // storage enabled constructor with default file name
    Collection(size_t vectorSize, bool enableStorage = false) 
    : storageEnabled(enableStorage), vecSize(vectorSize) {
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
    Collection(size_t vectorSize, const string& filename) 
    : storageEnabled(true), vecSize(vectorSize) {
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
        if (doc.getEmbedding().size() != vecSize) {
            throw std::runtime_error("Document embedding size does not match collection vector size");
        }
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
        if (doc.getEmbedding().size() != vecSize) {
            throw std::runtime_error("Document embedding size does not match collection vector size");
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

    const Document& getDocument(size_t id) const {
        if (id < documents.size()) {
            return documents[id];
        }
        throw std::out_of_range("Document ID out of range");
    }

    // Method to get the size of the collection
    size_t size() const {
        return documents.size();
    }

    // Method to get the vector size
    size_t getVectorSize() const {
        return vecSize;
    }
};
#include "vector_store.hpp"

// Default constructor without persistence
vector_store::vector_store() 
    : collection(make_shared<Collection>()), 
    similarity(make_shared<CosineSimilarity>()),
    index(make_shared<FlatIndex>(collection, similarity)) 
    {
        cout << "default vector store created!!\n";
    }

// Custom setup constructor
vector_store::vector_store(shared_ptr<Index> ind, shared_ptr<Collection> coll)
    : collection(coll), index(ind) {}

// Method to insert a document into the vector store
size_t vector_store::insert(Document& doc) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
        return -1; // Indicate failure
    }
    return index->insert(doc);
}

// Method to update a document in the vector store
void vector_store::update(size_t id, Document& doc) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
        return;
    }
    index->update(id, doc);
}

// Method to insert multiple documents into the vector store
vector<size_t> vector_store::insert(vector<Document>& docs){
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
        return {};
    }
    vector<size_t> res;
    for (auto& doc: docs){
        res.push_back(insert(doc));
    }
    return res;
}

// Method to search for k documents of a specific metadata
vector<Document> vector_store::search(const Metadata& meta, size_t k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->search(meta, k);
}

// Method to search for top k similar documents by embedding
vector<Document> vector_store::search(const vector<float>& embedding, size_t k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->search(embedding, k);
}

// Method to search for similar documents and their scores by embedding
vector<std::pair<float, Document>> vector_store::searchWithScores(const vector<float>& embedding, size_t k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->searchWithScores(embedding, k);
}

// Method to search for top k similar documents by embedding with same metadata
vector<Document> vector_store::search(const Metadata& meta, const vector<float>& embedding, size_t k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->search(meta, embedding, k);
}

// Method to search for similar documents and their scores by embedding with same metadata
vector<std::pair<float, Document>> vector_store::searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    return index->searchWithScores(meta, embedding, k);
}

// Method to fetch id of a document by metadata
size_t vector_store::fetchId(const Metadata& meta) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    for (size_t i=0; i<collection->size(); ++i) {
        if (collection->getDocuments()[i].getMetadata() == meta) {
            return i;
        }
    }
    return -1; // Indicate not found
}

// Method to fetch id of a document by embedding
size_t vector_store::fetchId(const vector<float>& embedding) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    for (size_t i=0; i<collection->size(); ++i) {
        if (collection->getDocuments()[i].getEmbedding() == embedding) {
            return i;
        }
    }
    return -1; // Indicate not found
}

// Method to fetch document by id
Document vector_store::fetchDocument(size_t id) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    if (id >= collection->size()) {
        throw std::out_of_range("Document ID out of range.");
    }
    return collection->getDocuments()[id];
}

// Method to fetch id of a document
size_t vector_store::fetchId(const Document& doc) {
    if (!index) {
        throw std::runtime_error("Index is not initialized.");
    }
    for (size_t i=0; i<collection->size(); ++i) {
        if (collection->getDocuments()[i] == doc) {
            return i;
        }
    }
    return -1; // Indicate not found
}

// Method to get the collection
shared_ptr<Collection> vector_store::getCollection() {
    return collection;
}
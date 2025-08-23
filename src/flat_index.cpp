#include "index.hpp"
#include <algorithm>

// Constructor to initialize the FlatIndex with a collection and similarity measure
FlatIndex::FlatIndex(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
    : Index(coll, sim) {
    cout << ">> flat index created!!\n";
}

// Method to insert a document into the index
size_t FlatIndex::insert(Document& doc) {
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    return collection->insert(doc);
}


// Method to update a document in the index
void FlatIndex::update(size_t id, Document& doc){
    if (!collection){
        throw std::runtime_error("collection is null");
    }
    collection->update(id, doc);
}

// Method to search for documents of a specific metadata
vector<Document> FlatIndex::search(const Metadata& meta, size_t k) const {
    vector<Document> results;
    // Iterate through the collection and find documents matching the metadata
    for (const auto& doc : collection->getDocuments()) {
        if (doc.getMetadata() == meta) {
            results.emplace_back(doc);
            if (results.size() >= k) {
                break; // Stop if we have enough results
            }
        }
    }
    return results;
}

// Method to search for similar documents by embedding
vector<Document> FlatIndex::search(const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {}; // Return empty vector if k is 0 or collection is empty
    }

    // Create a vector to store pairs of similarity scores and documents
    vector<pair<float, Document>> results;

    // Iterate through the collection and compute similarity scores
    for (const auto& doc : collection->getDocuments()) {
        results.emplace_back(similarity->compute(doc.getEmbedding(), embedding), doc);
    }
    k = std::min(k, results.size());
    std::partial_sort(results.begin(), results.begin() + k, results.end(), [&](const auto& a, const auto& b){
        return a.first > b.first;
    });
    vector<Document> res(k);
    for (size_t i=0; i<k; i++){
        res[i] = results[i].second;
    }
    return res;
}

// Method to search for similar documents and their scores by embedding
vector<pair<float, Document>> FlatIndex::searchWithScores(const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {}; // Return empty vector if k is 0 or collection is empty
    }

    // Create a vector to store pairs of similarity scores and documents
    vector<pair<float, Document>> results;

    // Iterate through the collection and compute similarity scores
    for (const auto& doc : collection->getDocuments()) {
        results.emplace_back(similarity->compute(doc.getEmbedding(), embedding), doc);
    }

    k = std::min(k, results.size());

    std::partial_sort(results.begin(), results.begin() + k, results.end(), [&](const auto& a, const auto& b){
        return a.first > b.first;
    });

    // Resize the results vector to contain only the top k results
    results.resize(k);

    return results;
}

// Method to search for top k similar documents by embedding with same metadata
vector<Document> FlatIndex::search(const Metadata& meta, const vector<float>& embedding, size_t k) const {
    if (k == 0 || collection->size() == 0) {
        return {}; // Return empty vector if k is 0 or collection is empty
    }

    // Create a vector to store pairs of similarity scores and documents
    vector<pair<float, Document>> results;

    // Iterate through the collection and compute similarity scores
    for (const auto& doc : collection->getDocuments()) {
        if (doc.getMetadata() == meta) {
            // Compute similarity score and store the document
            results.emplace_back(similarity->compute(doc.getEmbedding(), embedding), doc);
        }
        // Skip documents that do not match the metadata
    }
    k = std::min(k, results.size());
    std::partial_sort(results.begin(), results.begin() + k, results.end(), [&](const auto& a, const auto& b){
        return a.first > b.first;
    });
    vector<Document> res(k);
    for (size_t i=0; i<k; i++){
        res[i] = results[i].second;
    }
    return res;
}

// Method to search for top k similar documents and their scores by embedding with same metadata
vector<pair<float, Document>> FlatIndex::searchWithScores(const Metadata& meta, const vector<float>& embedding, size_t k) const {
    vector<pair<float, Document>> results;
    // Iterate through the collection and find documents matching the metadata
    for (const auto& doc : collection->getDocuments()) {
        if (doc.getMetadata() == meta) {
            float score = similarity->compute(doc.getEmbedding(), embedding);
            if (results.size() < k || score > results.back().first) {
                results.emplace_back(score, doc);
                // Sort results based on similarity score
                sort(results.begin(), results.end(), 
                            [](const auto& a, const auto& b) {
                                return a.first > b.first;
                            });
                if (results.size() > k) {
                    results.pop_back();
                }
            }
        }
    }
    return results;
}
#pragma once
#include "collection.hpp"
#include "similarity.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

using std::pair;

class LSH {
protected:
    shared_ptr<Collection> collection;
    shared_ptr<Similarity> similarity;
public:
    virtual ~LSH() = default;
    
    // Constructor to initialize the LSH with a collection and similarity measure
    LSH(shared_ptr<Collection> coll, shared_ptr<Similarity> sim)
        : collection(coll), similarity(sim) {}
    
    // Method to insert a document into the LSH
    virtual void insert(const vector<float>& embedding, size_t docId) = 0;

    // Method to update a document in the LSH
    virtual void update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) = 0;

    // Method to search for top k similar documents by embedding
    virtual vector<Document> search(const vector<float>& query, size_t k) const = 0;

    // Method to search for similar documents and their scores by embedding
    virtual vector<pair<float, Document>> searchWithScores(const vector<float>& query, size_t k) const = 0;

    // Method to search for top k similar documents by embedding with same metadata
    virtual vector<Document> search(const Metadata& meta, const vector<float>& query, size_t k) const = 0;

    // Method to search for similar documents and their scores by embedding with same metadata
    virtual vector<pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const = 0;
};

// Custom hash function for band keys
struct BandKeyHash {
    size_t operator()(const vector<int>& k) const {
        // Simple hash combine: shift + XOR
        size_t hash = 0;
        for (const auto& elem : k) {
            hash = (hash << 1) ^ elem;
        }
        return hash;
    }
};








// For Jaccard
class LSHJaccard: public LSH {
private:
    size_t numHashFunctions;
    size_t numBands;
    size_t rowsPerBand;
    size_t vecSize = 0; // Size of the vectors in the collection

    // Utility function to binarize an embedding
    vector<int> binarize(const vector<float>& embedding) const;

    // Hash table to store the bands and their corresponding document IDs
    vector<unordered_map<vector<int>, std::unordered_set<size_t>, BandKeyHash>> bands;

    // Hash functions
    vector<vector<int>> hashFunctions;
    
public:
    LSHJaccard(size_t nHashFunctions, size_t nBands, size_t rowsPerBand, shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    // Method to compute signature for a given embedding
    vector<int> computeSignature(const vector<int>& embedding) const;

    // Method to hash a document's embedding into bands
    void insert(const vector<float>& embedding, size_t docId) override;

    // Method to update a document's embedding in the LSH
    void update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) override;

    // Method to search for similar documents based on a query embedding
    vector<Document> search(const vector<float>& query, size_t k) const override;

    // Method to search for similar documents and their scores by embedding
    vector<pair<float, Document>> searchWithScores(const vector<float>& query, size_t k) const override;

    // Method to search for top k similar documents by embedding with same metadata
    vector<Document> search(const Metadata& meta, const vector<float>& query, size_t k) const override;

    // Method to search for similar documents and their scores by embedding with same metadata
    vector<pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const override; 
};







// For Cosine
class LSHCosine: public LSH {
private:
    size_t numBits;
    size_t numBands;
    size_t vecSize = 0; // Size of the vectors in the collection
    
    // Hash table to store the bands and their corresponding document IDs
    vector<unordered_map<vector<int>, std::unordered_set<size_t>, BandKeyHash>> bands;

    // Hash functions
    vector<vector<vector<float>>> hashFunctions;

public:
    LSHCosine(size_t nBits, size_t nBands, shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    // Method to compute signature for a given embedding
    vector<vector<int>> computeSignature(const vector<float>& embedding) const;

    // Method to hash a document's embedding into bands
    void insert(const vector<float>& embedding, size_t docId) override;

    // Method to update a document's embedding in the LSH
    void update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) override;

    // Method to search for similar documents based on a query embedding
    vector<Document> search(const vector<float>& query, size_t k) const override;

    // Method to search for similar documents and their scores by embedding
    vector<pair<float, Document>> searchWithScores(const vector<float>& query, size_t k) const override;

    // Method to search for top k similar documents by embedding with same metadata
    vector<Document> search(const Metadata& meta, const vector<float>& query, size_t k) const override;

    // Method to search for similar documents and their scores by embedding with same metadata
    vector<pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const override; 
};










// For Cosine
class LSHEuclidean: public LSH {
private:
    size_t numBits;   // Number of Hash Functions per Hash Table
    size_t numBands;   // Number of Hash Tables
    size_t bucketWidth;   // Bucket Width
    size_t vecSize = 0; // Size of the vectors in the collection

    // Hash table to store the bands and their corresponding document IDs
    vector<unordered_map<vector<int>, std::unordered_set<size_t>, BandKeyHash>> bands;

    // Hash functions and random offset
    vector<vector<vector<float>>> hashFunctions;
    vector<vector<float>> offsets;

public:
    LSHEuclidean(size_t nBits, size_t nBands, size_t bWidth, shared_ptr<Collection> coll, shared_ptr<Similarity> sim);

    // Method to compute signature for a given embedding
    vector<vector<int>> computeSignature(const vector<float>& embedding) const;

    // Method to hash a document's embedding into bands
    void insert(const vector<float>& embedding, size_t docId) override;

    // Method to update a document's embedding in the LSH
    void update(size_t docId, const vector<float>& newEmbedding, const vector<float>& oldEmbedding) override;

    // Method to search for similar documents based on a query embedding
    vector<Document> search(const vector<float>& query, size_t k) const override;

    // Method to search for similar documents and their scores by embedding
    vector<pair<float, Document>> searchWithScores(const vector<float>& query, size_t k) const override;

    // Method to search for top k similar documents by embedding with same metadata
    vector<Document> search(const Metadata& meta, const vector<float>& query, size_t k) const override;

    // Method to search for similar documents and their scores by embedding with same metadata
    vector<pair<float, Document>> searchWithScores(const Metadata& meta, const vector<float>& query, size_t k) const override; 
};






// Utility function to make lsh
shared_ptr<LSH> makeLSH(shared_ptr<Collection>& coll, shared_ptr<Similarity>& sim);
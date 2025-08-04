#pragma once
#include "vector_store.hpp"
#include "similarity.hpp"
#include "metadata.hpp"
#include "metadata_filter.hpp"
#include <vector>
#include <utility>
#include <unordered_map>
#include <random>
#include <unordered_map>
#include <random>

// namespace
using std::vector;
using std::pair;
using std::unordered_map;
using std::string;

class SearchEngine {
protected:
    const VectorStore& store;
    const SimilarityMetric& metric;

public:
    SearchEngine(const VectorStore& store, const SimilarityMetric& metric):
        store(store), metric(metric) {}

    // Returns top-K results as {score, vector}
    virtual vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, json filter = json{}) const = 0;
};

class FlatSearchEngine : public SearchEngine {
public:
    FlatSearchEngine(const VectorStore& store, const SimilarityMetric& metric);

    vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, json filter = json{}) const override;
};

class LSHSearchEngine : public SearchEngine {
private:
    struct LSHHash {
        vector<float> random_vector;
        float threshold;
        
        LSHHash(size_t dim) : random_vector(dim), threshold(0.0f) {
            // Initialize with random values from normal distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            
            for (auto& val : random_vector) {
                val = dist(gen);
            }
        }
        
        size_t hash(const vector<float>& vec) const {
            float dot_product = 0.0f;
            for (size_t i = 0; i < vec.size(); ++i) {
                dot_product += vec[i] * random_vector[i];
            }
            return (dot_product > threshold) ? 1 : 0;
        }
    };
    
    vector<vector<LSHHash>> hash_tables;
    vector<unordered_map<size_t, vector<size_t>>> buckets; // hash -> document indices
    size_t num_tables;
    size_t num_hashes_per_table;
    
    void buildHashTables();
    size_t computeHashSignature(const vector<float>& vec, size_t table_idx) const;

public:
    LSHSearchEngine(const VectorStore& store, const SimilarityMetric& metric, 
                   size_t num_tables = 10, size_t num_hashes_per_table = 8);
    
    vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, json filter = json{}) const override;
};
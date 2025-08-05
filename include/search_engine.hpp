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

class HNSWSearchEngine : public SearchEngine {
private:
    struct HNSWNode {
        size_t doc_idx;
        vector<float> embedding;
        vector<vector<size_t>> connections; // connections[layer] = list of neighbor indices
        
        HNSWNode(size_t idx, const vector<float>& vec) : doc_idx(idx), embedding(vec) {}
    };
    
    vector<HNSWNode> nodes;
    vector<size_t> entry_points; // entry point for each layer
    size_t max_layers;
    size_t max_connections_per_layer;
    size_t ef_construction; // size of dynamic candidate list during construction
    size_t ef_search; // size of dynamic candidate list during search
    double level_multiplier;
    
    // Helper methods
    size_t getRandomLevel() const;
    vector<size_t> searchLayer(const vector<float>& query, const vector<size_t>& entry_points, 
                              size_t num_closest, size_t layer) const;
    void selectNeighbors(vector<pair<float, size_t>>& candidates, size_t max_connections) const;
    float computeDistance(const vector<float>& a, const vector<float>& b) const;
    void insertNode(size_t node_idx, size_t target_layer);
    
public:
    HNSWSearchEngine(const VectorStore& store, const SimilarityMetric& metric,
                    size_t max_connections = 16, size_t ef_construction = 200, size_t ef_search = 50);
    
    vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, json filter = json{}) const override;
};

class AnnoySearchEngine : public SearchEngine {
private:
    struct AnnoyNode {
        vector<float> hyperplane;
        float hyperplane_offset;
        size_t left_child;
        size_t right_child;
        vector<size_t> document_indices; // For leaf nodes
        bool is_leaf;
        
        AnnoyNode() : hyperplane_offset(0.0f), left_child(SIZE_MAX), right_child(SIZE_MAX), is_leaf(false) {}
    };
    
    vector<AnnoyNode> nodes; // All nodes in a flat structure
    vector<size_t> tree_roots; // Root node index for each tree
    size_t num_trees;
    size_t max_leaf_size;
    size_t dimensions;
    
    // Helper methods
    size_t buildTreeRecursive(const vector<size_t>& doc_indices, size_t depth = 0);
    void searchTreeRecursive(size_t node_idx, const vector<float>& query, 
                           vector<size_t>& candidates, size_t max_candidates) const;
    vector<float> generateRandomHyperplane() const;
    float computeDotProduct(const vector<float>& a, const vector<float>& b) const;
    void splitByHyperplane(const vector<size_t>& doc_indices, const vector<float>& hyperplane,
                          float offset, vector<size_t>& left, vector<size_t>& right) const;
    
public:
    AnnoySearchEngine(const VectorStore& store, const SimilarityMetric& metric,
                     size_t num_trees = 10, size_t max_leaf_size = 50);
    
    vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, json filter = json{}) const override;
};
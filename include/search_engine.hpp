#pragma once
#include "vector_store.hpp"
#include "similarity.hpp"
#include <vector>
#include <utility>

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
    virtual vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, unordered_map<string, variant<string, int>> filter = {}) const = 0;
};

class FlatSearchEngine : public SearchEngine {
public:
    FlatSearchEngine(const VectorStore& store, const SimilarityMetric& metric);

    vector<pair<float, Document>> search(const vector<float>& query, size_t k = 5, unordered_map<string, variant<string, int>> filter = {}) const override;
};
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
private:
    const VectorStore& store;
    const SimilarityMetric& metric;

public:
    SearchEngine(const VectorStore& store, const SimilarityMetric& metric);

    // Returns top-K results as {score, vector}
    vector<pair<float, Vector>> search(const vector<float>& query, size_t k = 5) const;
};
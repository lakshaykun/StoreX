#include "include/search_engine.hpp"
#include <queue>
#include <algorithm>
// namespace
using std::vector;
using std::pair;
using std::min;
using std::partial_sort;
using std::unordered_map;
using std::string;

SearchEngine::SearchEngine(const VectorStore& store, const SimilarityMetric& metric)
    : store(store), metric(metric) {}

vector<pair<float, Vector>> SearchEngine::search(const vector<float>& query, size_t k) const {
    const auto& allVectors = store.getAll();

    // Pair of score and reference to Vector
    vector<pair<float, Vector>> results;

    for (const auto& v : allVectors) {
        float score = metric.compute(query, v.values);
        results.emplace_back(score, v);
    }

    // Get top-k sorted descending by score
    partial_sort(results.begin(), results.begin() + min(k, results.size()), results.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;  // Higher score = better match
        });

    if (results.size() > k)
        results.resize(k);

    return results;
}

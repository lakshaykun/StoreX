#include "search_engine.hpp"
#include <queue>
#include <algorithm>
// namespace
using std::vector;
using std::pair;
using std::min;
using std::partial_sort;
using std::unordered_map;
using std::string;

FlatSearchEngine::FlatSearchEngine(const VectorStore& store, const SimilarityMetric& metric)
    : SearchEngine(store, metric) {}

vector<pair<float, Document>> FlatSearchEngine::search(const vector<float>& query, size_t k, unordered_map<string, variant<string, int>> filter) const {
    vector<pair<float, Document>> results;
    const auto& documents = store.getAll();

    for (const auto& doc : documents) {
        // Apply filter if provided
        if (!filter.empty()) {
            bool matches = true;
            for (const auto& [key, value] : filter) {
                auto it = doc.metadata.find(key);
                if (it == doc.metadata.end() || it->second != value) {
                    matches = false;
                    break;
                }
            }
            if (!matches) continue;
        }

        float score = metric.compute(query, doc.embedding);
        results.emplace_back(score, doc);
    }

    // Sort results by score in descending order
    partial_sort(results.begin(), results.begin() + min(k, results.size()), results.end(),
                  [](const pair<float, Document>& a, const pair<float, Document>& b) {
                      return a.first > b.first;
                  });

    // Resize to top-k results
    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}
#include "search_engine.hpp"
#include "metadata.hpp"
#include "metadata_filter.hpp"
#include "json.hpp"
#include <queue>
#include <algorithm>
#include <iostream>
// namespace
using namespace std;
using json = nlohmann::json;

// Flat search engine implementation
FlatSearchEngine::FlatSearchEngine(const VectorStore& store, const SimilarityMetric& metric)
    : SearchEngine(store, metric) {}

vector<pair<float, Document>> FlatSearchEngine::search(const vector<float>& query, size_t k, json filter) const {
    vector<pair<float, Document>> results;
    const auto& documents = store.getAll();
    
    if (filter.empty()){
        for (const auto& doc : documents) {
            float score = metric.compute(query, doc.embedding);
            results.emplace_back(score, doc);
        }
    } else {
        Filter parsedFilter;
        try {
            parsedFilter = parseFilter(filter);
        } catch (const invalid_argument& e) {
            cerr << "Invalid filter: " << e.what() << endl;
            return results; // Return empty results on invalid filter
        }

        for (const auto& doc : documents) {
            // Apply filter if provided
            if (!evaluate(doc.metadata, parsedFilter)) continue;

            float score = metric.compute(query, doc.embedding);
            results.emplace_back(score, doc);
        }
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
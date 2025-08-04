#include "search_engine.hpp"
#include "metadata.hpp"
#include "metadata_filter.hpp"
#include "json.hpp"
#include <queue>
#include <algorithm>
#include <iostream>
#include <unordered_set>

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

// LSH search engine implementation
LSHSearchEngine::LSHSearchEngine(const VectorStore& store, const SimilarityMetric& metric,
                                size_t num_tables, size_t num_hashes_per_table)
    : SearchEngine(store, metric), num_tables(num_tables), num_hashes_per_table(num_hashes_per_table) {
    
    // Initialize hash tables
    const auto& documents = store.getAll();
    if (!documents.empty()) {
        size_t dim = documents[0].embedding.size();
        hash_tables.resize(num_tables);
        buckets.resize(num_tables);
        
        for (size_t i = 0; i < num_tables; ++i) {
            hash_tables[i].resize(num_hashes_per_table, LSHHash(dim));
        }
        
        buildHashTables();
    }
}

void LSHSearchEngine::buildHashTables() {
    const auto& documents = store.getAll();
    
    for (size_t doc_idx = 0; doc_idx < documents.size(); ++doc_idx) {
        const auto& doc = documents[doc_idx];
        
        for (size_t table_idx = 0; table_idx < num_tables; ++table_idx) {
            size_t hash_signature = computeHashSignature(doc.embedding, table_idx);
            buckets[table_idx][hash_signature].push_back(doc_idx);
        }
    }
}

size_t LSHSearchEngine::computeHashSignature(const vector<float>& vec, size_t table_idx) const {
    size_t signature = 0;
    for (size_t hash_idx = 0; hash_idx < num_hashes_per_table; ++hash_idx) {
        size_t bit = hash_tables[table_idx][hash_idx].hash(vec);
        signature |= (bit << hash_idx);
    }
    return signature;
}

vector<pair<float, Document>> LSHSearchEngine::search(const vector<float>& query, size_t k, json filter) const {
    std::unordered_set<size_t> candidate_indices;
    const auto& documents = store.getAll();
    
    // Collect candidates from all hash tables
    for (size_t table_idx = 0; table_idx < num_tables; ++table_idx) {
        size_t query_hash = computeHashSignature(query, table_idx);
        
        auto bucket_it = buckets[table_idx].find(query_hash);
        if (bucket_it != buckets[table_idx].end()) {
            for (size_t doc_idx : bucket_it->second) {
                candidate_indices.insert(doc_idx);
            }
        }
    }
    
    // Compute actual similarities for candidates
    vector<pair<float, Document>> results;
    results.reserve(candidate_indices.size());
    
    if (filter.empty()) {
        for (size_t doc_idx : candidate_indices) {
            const auto& doc = documents[doc_idx];
            float similarity = metric.compute(query, doc.embedding);
            results.emplace_back(similarity, doc);
        }
    } else {
        Filter parsedFilter;
        try {
            parsedFilter = parseFilter(filter);
        } catch (const invalid_argument& e) {
            cerr << "Invalid filter: " << e.what() << endl;
            return results; // Return empty results on invalid filter
        }

        for (size_t doc_idx : candidate_indices) {
            const auto& doc = documents[doc_idx];
            
            // Apply metadata filter if provided
            if (!evaluate(doc.metadata, parsedFilter)) {
                continue;
            }
            
            float similarity = metric.compute(query, doc.embedding);
            results.emplace_back(similarity, doc);
        }
    }
    
    // Sort by similarity (descending) and return top-k
    std::partial_sort(results.begin(), 
                     results.begin() + std::min(k, results.size()), 
                     results.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

#include "search_engine.hpp"
#include "metadata.hpp"
#include "metadata_filter.hpp"
#include "json.hpp"
#include <queue>
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <cmath>
#include <climits>
#include <numeric>
#include <thread>

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

// HNSW search engine implementation
HNSWSearchEngine::HNSWSearchEngine(const VectorStore& store, const SimilarityMetric& metric,
                                  size_t max_connections, size_t ef_construction, size_t ef_search)
    : SearchEngine(store, metric), max_connections_per_layer(max_connections), 
      ef_construction(ef_construction), ef_search(ef_search), level_multiplier(1.0 / log(2.0)) {
    
    const auto& documents = store.getAll();
    if (documents.empty()) return;
    
    max_layers = static_cast<size_t>(log2(documents.size())) + 1;
    entry_points.resize(max_layers, SIZE_MAX);
    
    // Build HNSW graph
    nodes.reserve(documents.size());
    
    for (size_t i = 0; i < documents.size(); ++i) {
        nodes.emplace_back(i, documents[i].embedding);
        nodes[i].connections.resize(max_layers);
        
        size_t target_layer = getRandomLevel();
        insertNode(i, target_layer);
    }
}

size_t HNSWSearchEngine::getRandomLevel() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    
    size_t level = 0;
    while (dis(gen) < 0.5 && level < max_layers - 1) {
        level++;
    }
    return level;
}

float HNSWSearchEngine::computeDistance(const vector<float>& a, const vector<float>& b) const {
    // Convert similarity to distance (assuming cosine similarity)
    float similarity = metric.compute(a, b);
    return 1.0f - similarity; // Convert to distance
}

vector<size_t> HNSWSearchEngine::searchLayer(const vector<float>& query, const vector<size_t>& entry_points, 
                                            size_t num_closest, size_t layer) const {
    if (entry_points.empty()) return {};
    
    // Priority queue for candidates (min-heap by distance)
    priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, greater<pair<float, size_t>>> candidates;
    // Priority queue for results (max-heap by distance)
    priority_queue<pair<float, size_t>> results;
    unordered_set<size_t> visited;
    
    // Initialize with entry points
    for (size_t ep : entry_points) {
        if (ep == SIZE_MAX || ep >= nodes.size()) continue;
        float dist = computeDistance(query, nodes[ep].embedding);
        candidates.push({dist, ep});
        results.push({dist, ep});
        visited.insert(ep);
    }
    
    while (!candidates.empty()) {
        auto [current_dist, current_node] = candidates.top();
        candidates.pop();
        
        // Stop if current distance is worse than the worst in results and we have enough
        if (results.size() >= num_closest && current_dist > results.top().first) {
            break;
        }
        
        // Explore neighbors
        if (layer < nodes[current_node].connections.size()) {
            for (size_t neighbor : nodes[current_node].connections[layer]) {
                if (neighbor >= nodes.size() || visited.count(neighbor)) continue;
                
                visited.insert(neighbor);
                float neighbor_dist = computeDistance(query, nodes[neighbor].embedding);
                
                if (results.size() < num_closest || neighbor_dist < results.top().first) {
                    candidates.push({neighbor_dist, neighbor});
                    results.push({neighbor_dist, neighbor});
                    
                    if (results.size() > num_closest) {
                        results.pop();
                    }
                }
            }
        }
    }
    
    // Extract results
    vector<size_t> closest;
    closest.reserve(results.size());
    while (!results.empty()) {
        closest.push_back(results.top().second);
        results.pop();
    }
    reverse(closest.begin(), closest.end()); // Best first
    return closest;
}

void HNSWSearchEngine::selectNeighbors(vector<pair<float, size_t>>& candidates, size_t max_connections) const {
    if (candidates.size() <= max_connections) return;
    
    // Simple heuristic: keep the closest ones
    sort(candidates.begin(), candidates.end());
    candidates.resize(max_connections);
}

void HNSWSearchEngine::insertNode(size_t node_idx, size_t target_layer) {
    if (nodes.empty() || node_idx >= nodes.size()) return;
    
    // Insert from top layer down to layer 0
    vector<size_t> current_closest;
    
    // Start from the highest layer where we have an entry point
    for (int layer = static_cast<int>(max_layers) - 1; layer >= 0; --layer) {
        size_t ul = static_cast<size_t>(layer);
        
        if (entry_points[ul] == SIZE_MAX) {
            // No entry point for this layer yet
            if (ul <= target_layer) {
                entry_points[ul] = node_idx;
            }
            continue;
        }
        
        // Search in this layer
        vector<size_t> entry_for_layer = {entry_points[ul]};
        if (!current_closest.empty()) {
            entry_for_layer = current_closest;
        }
        
        size_t ef = (ul > target_layer) ? 1 : ef_construction;
        current_closest = searchLayer(nodes[node_idx].embedding, entry_for_layer, ef, ul);
        
        // Connect to neighbors if we're at or below target layer
        if (ul <= target_layer) {
            vector<pair<float, size_t>> candidates;
            for (size_t neighbor : current_closest) {
                if (neighbor != node_idx) {
                    float dist = computeDistance(nodes[node_idx].embedding, nodes[neighbor].embedding);
                    candidates.emplace_back(dist, neighbor);
                }
            }
            
            selectNeighbors(candidates, max_connections_per_layer);
            
            // Add bidirectional connections
            for (const auto& [dist, neighbor] : candidates) {
                nodes[node_idx].connections[ul].push_back(neighbor);
                nodes[neighbor].connections[ul].push_back(node_idx);
                
                // Prune neighbor's connections if needed
                if (nodes[neighbor].connections[ul].size() > max_connections_per_layer) {
                    vector<pair<float, size_t>> neighbor_candidates;
                    for (size_t conn : nodes[neighbor].connections[ul]) {
                        if (conn != neighbor) {
                            float d = computeDistance(nodes[neighbor].embedding, nodes[conn].embedding);
                            neighbor_candidates.emplace_back(d, conn);
                        }
                    }
                    selectNeighbors(neighbor_candidates, max_connections_per_layer);
                    
                    nodes[neighbor].connections[ul].clear();
                    for (const auto& [d, conn] : neighbor_candidates) {
                        nodes[neighbor].connections[ul].push_back(conn);
                    }
                }
            }
        }
    }
}

vector<pair<float, Document>> HNSWSearchEngine::search(const vector<float>& query, size_t k, json filter) const {
    if (nodes.empty()) return {};
    
    const auto& documents = store.getAll();
    vector<size_t> current_closest;
    
    // Search from top layer down to layer 1
    for (int layer = static_cast<int>(max_layers) - 1; layer >= 1; --layer) {
        size_t ul = static_cast<size_t>(layer);
        if (entry_points[ul] == SIZE_MAX) continue;
        
        vector<size_t> entry_for_layer = current_closest.empty() ? 
            vector<size_t>{entry_points[ul]} : current_closest;
        current_closest = searchLayer(query, entry_for_layer, 1, ul);
    }
    
    // Search layer 0 with ef_search
    if (entry_points[0] != SIZE_MAX) {
        vector<size_t> entry_for_layer = current_closest.empty() ? 
            vector<size_t>{entry_points[0]} : current_closest;
        current_closest = searchLayer(query, entry_for_layer, max(k, ef_search), 0);
    }
    
    // Convert to results with similarity scores and apply filtering
    vector<pair<float, Document>> results;
    results.reserve(current_closest.size());
    
    Filter parsedFilter;
    bool has_filter = !filter.empty();
    if (has_filter) {
        try {
            parsedFilter = parseFilter(filter);
        } catch (const invalid_argument& e) {
            cerr << "Invalid filter: " << e.what() << endl;
            return results;
        }
    }
    
    for (size_t node_idx : current_closest) {
        if (node_idx >= nodes.size() || nodes[node_idx].doc_idx >= documents.size()) continue;
        
        const auto& doc = documents[nodes[node_idx].doc_idx];
        
        // Apply filter if provided
        if (has_filter && !evaluate(doc.metadata, parsedFilter)) continue;
        
        float similarity = metric.compute(query, doc.embedding);
        results.emplace_back(similarity, doc);
    }
    
    // Sort by similarity (descending) and limit to k
    sort(results.begin(), results.end(), 
         [](const auto& a, const auto& b) { return a.first > b.first; });
    
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

// Annoy search engine implementation
AnnoySearchEngine::AnnoySearchEngine(const VectorStore& store, const SimilarityMetric& metric,
                                    size_t num_trees, size_t max_leaf_size)
    : SearchEngine(store, metric), num_trees(num_trees), max_leaf_size(max_leaf_size) {
    
    const auto& documents = store.getAll();
    if (documents.empty()) return;
    
    dimensions = documents[0].embedding.size();
    tree_roots.reserve(num_trees);
    
    // Build multiple random projection trees
    vector<size_t> all_indices(documents.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    for (size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        size_t root_idx = buildTreeRecursive(all_indices);
        tree_roots.push_back(root_idx);
    }
}

vector<float> AnnoySearchEngine::generateRandomHyperplane() const {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
    
    vector<float> hyperplane(dimensions);
    for (auto& val : hyperplane) {
        val = dist(gen);
    }
    
    // Normalize the hyperplane
    float norm = 0.0f;
    for (float val : hyperplane) {
        norm += val * val;
    }
    norm = sqrt(norm);
    
    if (norm > 1e-10f) {
        for (auto& val : hyperplane) {
            val /= norm;
        }
    }
    
    return hyperplane;
}

float AnnoySearchEngine::computeDotProduct(const vector<float>& a, const vector<float>& b) const {
    float result = 0.0f;
    for (size_t i = 0; i < min(a.size(), b.size()); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void AnnoySearchEngine::splitByHyperplane(const vector<size_t>& doc_indices, 
                                         const vector<float>& hyperplane, float offset,
                                         vector<size_t>& left, vector<size_t>& right) const {
    const auto& documents = store.getAll();
    
    for (size_t doc_idx : doc_indices) {
        if (doc_idx >= documents.size()) continue;
        
        float dot_product = computeDotProduct(documents[doc_idx].embedding, hyperplane);
        
        if (dot_product <= offset) {
            left.push_back(doc_idx);
        } else {
            right.push_back(doc_idx);
        }
    }
}

size_t AnnoySearchEngine::buildTreeRecursive(const vector<size_t>& doc_indices, size_t depth) {
    // Create current node
    size_t current_node_idx = nodes.size();
    nodes.emplace_back();
    AnnoyNode& current_node = nodes.back();
    
    // If we have few enough documents or reached max depth, make this a leaf
    if (doc_indices.size() <= max_leaf_size || depth > 20) {
        current_node.is_leaf = true;
        current_node.document_indices = doc_indices;
        return current_node_idx;
    }
    
    // Generate random hyperplane for splitting
    current_node.hyperplane = generateRandomHyperplane();
    
    // Compute hyperplane offset as median projection
    vector<float> projections;
    projections.reserve(doc_indices.size());
    
    const auto& documents = store.getAll();
    for (size_t doc_idx : doc_indices) {
        if (doc_idx < documents.size()) {
            float projection = computeDotProduct(documents[doc_idx].embedding, current_node.hyperplane);
            projections.push_back(projection);
        }
    }
    
    if (!projections.empty()) {
        sort(projections.begin(), projections.end());
        current_node.hyperplane_offset = projections[projections.size() / 2];
    }
    
    // Split documents by hyperplane
    vector<size_t> left_indices, right_indices;
    splitByHyperplane(doc_indices, current_node.hyperplane, 
                     current_node.hyperplane_offset, left_indices, right_indices);
    
    // Ensure both sides have at least one document
    if (left_indices.empty() || right_indices.empty()) {
        current_node.is_leaf = true;
        current_node.document_indices = doc_indices;
        return current_node_idx;
    }
    
    // Recursively build subtrees
    current_node.left_child = buildTreeRecursive(left_indices, depth + 1);
    current_node.right_child = buildTreeRecursive(right_indices, depth + 1);
    
    return current_node_idx;
}

void AnnoySearchEngine::searchTreeRecursive(size_t node_idx, const vector<float>& query,
                                           vector<size_t>& candidates, size_t max_candidates) const {
    if (node_idx >= nodes.size() || node_idx == SIZE_MAX) return;
    
    const AnnoyNode& node = nodes[node_idx];
    
    if (node.is_leaf) {
        // Add all documents in this leaf to candidates
        for (size_t doc_idx : node.document_indices) {
            if (candidates.size() < max_candidates) {
                candidates.push_back(doc_idx);
            }
        }
        return;
    }
    
    // Decide which side of the hyperplane the query is on
    float query_projection = computeDotProduct(query, node.hyperplane);
    bool go_left = query_projection <= node.hyperplane_offset;
    
    // Search the preferred side first
    if (go_left && node.left_child != SIZE_MAX) {
        searchTreeRecursive(node.left_child, query, candidates, max_candidates);
    } else if (!go_left && node.right_child != SIZE_MAX) {
        searchTreeRecursive(node.right_child, query, candidates, max_candidates);
    }
    
    // If we don't have enough candidates, search the other side too
    if (candidates.size() < max_candidates) {
        if (go_left && node.right_child != SIZE_MAX) {
            searchTreeRecursive(node.right_child, query, candidates, max_candidates);
        } else if (!go_left && node.left_child != SIZE_MAX) {
            searchTreeRecursive(node.left_child, query, candidates, max_candidates);
        }
    }
}

vector<pair<float, Document>> AnnoySearchEngine::search(const vector<float>& query, size_t k, json filter) const {
    if (tree_roots.empty()) return {};
    
    const auto& documents = store.getAll();
    unordered_set<size_t> candidate_set;
    
    // Search all trees and collect candidates
    size_t candidates_per_tree = max(k * 2, static_cast<size_t>(100));
    
    for (size_t root_idx : tree_roots) {
        vector<size_t> tree_candidates;
        searchTreeRecursive(root_idx, query, tree_candidates, candidates_per_tree);
        
        for (size_t candidate : tree_candidates) {
            candidate_set.insert(candidate);
        }
    }
    
    // Convert to results with similarity scores and apply filtering
    vector<pair<float, Document>> results;
    results.reserve(candidate_set.size());
    
    Filter parsedFilter;
    bool has_filter = !filter.empty();
    if (has_filter) {
        try {
            parsedFilter = parseFilter(filter);
        } catch (const invalid_argument& e) {
            cerr << "Invalid filter: " << e.what() << endl;
            return results;
        }
    }
    
    for (size_t doc_idx : candidate_set) {
        if (doc_idx >= documents.size()) continue;
        
        const auto& doc = documents[doc_idx];
        
        // Apply filter if provided
        if (has_filter && !evaluate(doc.metadata, parsedFilter)) continue;
        
        float similarity = metric.compute(query, doc.embedding);
        results.emplace_back(similarity, doc);
    }
    
    // Sort by similarity (descending) and limit to k
    sort(results.begin(), results.end(),
         [](const auto& a, const auto& b) { return a.first > b.first; });
    
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}

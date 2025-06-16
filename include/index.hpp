#pragma once
#include <string>
#include <vector>
// namespace
using std::string;
using std::vector;

// abstract class for indexing methods like lsh, hnsw, etc.
class IndexingMethod {
public:
    virtual ~IndexingMethod() = default;

    // Index the given data
    virtual void index(const vector<vector<float>>& data) = 0;

    // Search for the nearest neighbors of the query vector
    virtual vector<size_t> search(const vector<float>& query, size_t k) const = 0;

    // Get the name of the indexing method
    virtual string getName() const = 0;
};
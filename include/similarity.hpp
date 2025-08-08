#pragma once
#include <vector>
using std::vector;

// virtual class to compute similarity between vectors
class Similarity {
public:
    virtual ~Similarity() = default;

    // Method to compute the similarity score between two vectors
    virtual float compute(const vector<float>& emb1, const vector<float>& emb2) const = 0;
};
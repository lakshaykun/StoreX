#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>
#include "utility.hpp"

using std::vector;
using std::string;
using std::invalid_argument;
using std::shared_ptr;
using std::make_shared;

// virtual class to compute similarity between vectors
class Similarity {
public:
    virtual ~Similarity() = default;
    string name = "Similarity";
    // Method to compute the similarity score between two vectors
    virtual float compute(const vector<float>& emb1, const vector<float>& emb2) const = 0;
};

// declarations of similarity metrics in similarity.cpp

class CosineSimilarity : public Similarity {
public:
    CosineSimilarity() { name = "CosineSimilarity"; }
    float compute(const vector<float>& emb1, const vector<float>& emb2) const override;
};

class EuclideanSimilarity : public Similarity {
public:
    EuclideanSimilarity() { name = "EuclideanSimilarity"; }
    float compute(const vector<float>& emb1, const vector<float>& emb2) const override;
};

class JaccardSimilarity : public Similarity {
public:
    JaccardSimilarity() { name = "JaccardSimilarity"; }
    float compute(const vector<float>& emb1, const vector<float>& emb2) const override;
};

// Factory function to create a similarity object based on type
shared_ptr<Similarity> createSimilarity(const string& type);
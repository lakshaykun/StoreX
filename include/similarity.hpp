#pragma once
#include <vector>
#include <stdexcept>
using namespace std;

class SimilarityMetric {
public:
    virtual float compute(const vector<float>& a, const vector<float>& b) const = 0;
    virtual ~SimilarityMetric() = default;
};

class DotProductSimilarity : public SimilarityMetric {
public:
    float compute(const vector<float>& a, const vector<float>& b) const override;
};

class CosineSimilarity : public SimilarityMetric {
public:
    float compute(const vector<float>& a, const vector<float>& b) const override;
};

class EuclideanSimilarity : public SimilarityMetric {
public:
    float compute(const vector<float>& a, const vector<float>& b) const override;
};
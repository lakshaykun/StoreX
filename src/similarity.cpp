#include "similarity.hpp"
#include <cmath>
#include <stdexcept>
// namespace
using std::vector;
using std::invalid_argument;
using std::sqrt;
using std::exp;

float DotProductSimilarity::compute(const vector<float>& a, const vector<float>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must be of the same length.");
    }

    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

float CosineSimilarity::compute(const vector<float>& a, const vector<float>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must be of the same length.");
    }

    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    if (normA == 0.0f || normB == 0.0f) {
        return 0.0f;  // Or throw if zero-vector is not allowed
    }

    return dot / (sqrt(normA) * sqrt(normB));
}

float EuclideanSimilarity::compute(const vector<float>& a, const vector<float>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("Vectors must be of the same length.");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return exp(-sqrt(sum));  // Return exp(-distance) for similarity
}

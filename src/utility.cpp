#include "utility.hpp"

namespace utility {
    float dotProduct(const vector<float>& emb1, const vector<float>& emb2) {
        if (emb1.size() != emb2.size()) {
            throw std::invalid_argument("Vector dimensions must match");
        }
        
        return std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    }
}

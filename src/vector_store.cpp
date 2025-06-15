#include "include/vector_store.hpp"

void VectorStore::insert(const Vector& v) {
    vectors.push_back(v);
}

const vector<Vector>& VectorStore::getAll() const {
    return vectors;
}

#include "vector_store.hpp"

void VectorStore::insert(const Document& v) {
    Documents.push_back(v);
}

const vector<Document>& VectorStore::getAll() const {
    return Documents;
}

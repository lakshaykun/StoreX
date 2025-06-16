#pragma once
#include "document.hpp"
#include <vector>
using namespace std;

class VectorStore {
private:
    vector<Document> Documents;

public:
    void insert(const Document& v);
    const vector<Document>& getAll() const;
};

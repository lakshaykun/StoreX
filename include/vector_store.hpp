#pragma once
#include "vector.hpp"
#include <vector>
using namespace std;

class VectorStore {
private:
    vector<Vector> vectors;

public:
    void insert(const Vector& v);
    const vector<Vector>& getAll() const;
};

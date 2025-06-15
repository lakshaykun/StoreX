#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
using namespace std;

struct Vector {
    vector<float> values;
    unordered_map<string, variant<string, int>> metadata;

    Vector(const vector<float>& values,
           const unordered_map<string, variant<string, int>>& metadata = {})
        : values(values), metadata(metadata) {}
};

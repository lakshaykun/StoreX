#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
using namespace std;

struct Document {
    vector<float> embedding;
    unordered_map<string, variant<string, int>> metadata;

    Document(const vector<float>& embedding = {},
           const unordered_map<string, variant<string, int>>& metadata = {})
        : embedding(embedding), metadata(metadata) {}
};

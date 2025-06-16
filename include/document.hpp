#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include "metadata.hpp" 
using namespace std;

struct Document {
    vector<float> embedding;
    Metadata metadata;

    Document(const vector<float>& embedding = {}, const Metadata& metadata = {})
        : embedding(embedding), metadata(metadata) {}
};

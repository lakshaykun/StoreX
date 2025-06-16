// include/metadata.hpp
#pragma once
#include <string>
#include <unordered_map>
#include <variant>
using namespace std;

using MetadataValue = variant<string, int, float>;
using Metadata = unordered_map<string, MetadataValue>;
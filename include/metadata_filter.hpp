#pragma once
#include "metadata.hpp"
#include "document.hpp"
#include "json.hpp"
#include <vector>
using namespace std;
using json = nlohmann::json;

enum class Operator {
    EQ, // Equal
    NEQ, // Not Equal
    LT, // Less Than
    LTE, // Less Than or Equal
    GT, // Greater Than
    GTE, // Greater Than or Equal
    NIN, // Not In
    IN, // In
    AND, // And
    OR // Or
};

inline Operator parseOperator(const std::string& opStr) {
    if (opStr == "EQ") return Operator::EQ;
    if (opStr == "NEQ") return Operator::NEQ;
    if (opStr == "LT") return Operator::LT;
    if (opStr == "LTE") return Operator::LTE;
    if (opStr == "GT") return Operator::GT;
    if (opStr == "GTE") return Operator::GTE;
    if (opStr == "NIN") return Operator::NIN;
    if (opStr == "IN") return Operator::IN;
    if (opStr == "AND") return Operator::AND;
    if (opStr == "OR") return Operator::OR;
    throw std::invalid_argument("Unknown operator: " + opStr);
}

struct Filter {
    string field;
    Operator op;
    MetadataValue value;  // for EQ, LT, etc.
    vector<MetadataValue> values;  // for IN
    vector<Filter> children;  // for AND/OR
};

bool evaluate(const Metadata& metadata, const Filter& filter);

Filter parseFilter(const json& j);
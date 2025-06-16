#include "metadata_filter.hpp"
#include "metadata.hpp"
#include "json.hpp"
#include <variant>
#include <iostream>
using namespace std;

bool match(const MetadataValue& a, const MetadataValue& b) {
    // Check if both values are of the same type
    if (a.index() != b.index()) return false;
    return a == b;
}

bool evaluate(const Metadata& metadata, const Filter& filter) {
    // Check if the field exists in the metadata
    const auto& it = metadata.find(filter.field);
    if (it == metadata.end() && filter.op != Operator::AND && filter.op != Operator::OR) {
        return false;
    }
    const auto& value = it->second;

    switch (filter.op) {
        case Operator::EQ: return match(value, filter.value);
        case Operator::NEQ: return !match(value, filter.value);
        case Operator::LT:
        case Operator::LTE:
        case Operator::GT:
        case Operator::GTE: {
            if (value.index() != filter.value.index()) return false;
            if (auto* intVal = get_if<int>(&value)) {
                int comp = get<int>(filter.value);
                if (filter.op == Operator::LT) return *intVal < comp;
                if (filter.op == Operator::LTE) return *intVal <= comp;
                if (filter.op == Operator::GT) return *intVal > comp;
                if (filter.op == Operator::GTE) return *intVal >= comp;
            }
            if (auto* floatVal = get_if<float>(&value)) {
                float comp = get<float>(filter.value);
                if (filter.op == Operator::LT) return *floatVal < comp;
                if (filter.op == Operator::LTE) return *floatVal <= comp;
                if (filter.op == Operator::GT) return *floatVal > comp;
                if (filter.op == Operator::GTE) return *floatVal >= comp;
            }
            return false;
        }
        case Operator::IN: {
            for (const auto& val : filter.values) {
                if (match(value, val)) return true;
            }
            return false;
        }
        case Operator::NIN: {
            for (const auto& val : filter.values) {
                if (match(value, val)) return false;
            }
            return true;
        }
        case Operator::AND: {
            for (const auto& child : filter.children) {
                if (!evaluate(metadata, child)) return false;
            }
            return true;
        }
        case Operator::OR: {
            for (const auto& child : filter.children) {
                if (evaluate(metadata, child)) return true;
            }
            return false;
        }
    }
    return false;
}


Filter parseFilter(const json& j) {
    Filter f;
    f.op = parseOperator(j.at("op").get<std::string>());
    if (f.op == Operator::AND || f.op == Operator::OR) {
        for (const auto& child : j.at("children")) {
            f.children.push_back(parseFilter(child));
        }
    } else if (f.op == Operator::IN) {
        f.field = j.at("field").get<std::string>();
        for (const auto& v : j.at("values")) {
            if (v.is_number_integer()) f.values.push_back(v.get<int>());
            else if (v.is_number_float()) f.values.push_back(v.get<float>());
            else if (v.is_string()) f.values.push_back(v.get<std::string>());
        }
    } else {
        f.field = j.at("field").get<std::string>();
        const auto& val = j.at("value");
        if (val.is_number_integer()) f.value = val.get<int>();
        else if (val.is_number_float()) f.value = val.get<float>();
        else if (val.is_string()) f.value = val.get<std::string>();
    }

    return f;
}
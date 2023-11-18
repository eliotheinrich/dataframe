#include "DataSlide.h"
#include "utils.h"

using namespace dataframe_utils;

DataSlide::DataSlide(const std::string &s) {
    std::string trimmed = s;
    uint32_t start_pos = trimmed.find_first_not_of(" \t\n\r");
    uint32_t end_pos = trimmed.find_last_not_of(" \t\n\r");
    trimmed = trimmed.substr(start_pos, end_pos - start_pos + 1);

    nlohmann::json ds_json;
    if (trimmed.empty() || trimmed.front() != '{' || trimmed.back() != '}')
        ds_json = nlohmann::json::parse("{" + trimmed + "}");
    else
        ds_json = nlohmann::json::parse(trimmed);

    for (auto const &[k, val] : ds_json.items()) {
        if (val.type() == nlohmann::json::value_t::array) {
            add_data(k);
            for (auto const &v : val)
                push_data(k, Sample(v.dump()));
        } else
            add_param(k, parse_json_type(val));
    }
}

DataSlide::DataSlide(const DataSlide& other) {
    for (auto const& [key, val]: other.params)
        add_param(key, val);

    for (auto const& [key, vals] : other.data) {
        data[key] = std::vector<Sample>();
        for (auto const& val : vals)
            data[key].push_back(val);
    }
}

DataSlide DataSlide::copy_params(const DataSlide& other) {
    DataSlide slide;
    for (auto const& [key, val]: other.params)
        slide.add_param(key, val);

    return slide;
}


std::vector<double> DataSlide::get_data(const std::string& s) const {
    if (!data.count(s))
        return std::vector<double>();

    std::vector<double> d;
    for (auto const &s : data.at(s))
        d.push_back(s.get_mean());
    
    return d;
}

std::vector<double> DataSlide::get_std(const std::string& s) const {
    if (!data.count(s))
        return std::vector<double>();

    std::vector<double> d;
    for (auto const &s : data.at(s))
        d.push_back(s.get_std());

    return d;
}

bool DataSlide::remove(const std::string& s) {
    if (params.count(s)) { 
        return params.erase(s);
    } else if (data.count(s)) {
        data.erase(s);
        return true;
    }
    return false;
}

std::string DataSlide::to_string(uint32_t indentation, bool pretty, bool record_error) const {
    std::string tab = pretty ? "\t" : "";
    std::string nline = pretty ? "\n" : "";
    std::string tabs = "";
    for (uint32_t i = 0; i < indentation; i++) tabs += tab;
    
    std::string s = params_to_string(params, indentation);

    if ((!params.empty()) && (!data.empty())) s += "," + nline + tabs;

    std::string delim = "," + nline + tabs;
    std::vector<std::string> buffer;

    for (auto const &[key, samples] : data) {
        std::vector<std::string> sample_buffer;
        for (auto sample : samples) {
            sample_buffer.push_back(sample.to_string(record_error));
        }

        buffer.push_back("\"" + key + "\": [" + join(sample_buffer, ", ") + "]");
    }

    s += join(buffer, delim);
    return s;
}

bool DataSlide::congruent(const DataSlide &ds, const var_t_eq& equality_comparator) {
    if (!params_eq(params, ds.params, equality_comparator)) {
        return false;
    }

    for (auto const &[key, samples] : data) {
        if (!ds.data.count(key)) {
            std::cout << key << " not congruent.\n";
            return false;
        }
        if (ds.data.at(key).size() != data.at(key).size()) {
            std::cout << key << " not congruent.\n";
            return false;
        }
    }
    for (auto const &[key, val] : ds.data) {
        if (!data.count(key)) {
            std::cout << key << " not congruent.\n";
            return false;
        }
    }

    return true;
}

DataSlide DataSlide::combine(const DataSlide &ds, const var_t_eq& equality_comparator) {
    if (!congruent(ds, equality_comparator)) {
        std::stringstream ss;
        ss << "DataSlides not congruent.\n"; 
        ss << to_string() << "\n\n\n" << ds.to_string() << std::endl;
        throw std::invalid_argument(ss.str());
    }

    DataSlide dn(params); 

    for (auto const &[key, samples] : data) {
        dn.add_data(key);
        for (uint32_t i = 0; i < samples.size(); i++)
            dn.push_data(key, samples[i].combine(ds.data.at(key)[i]));
    }

    return dn;
}

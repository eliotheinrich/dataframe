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
                push_data(k, v.dump());
        } else
            add_param(k, parse_json_type(val));
    }
}

DataSlide::DataSlide(const DataSlide& other) {
    for (auto const& [key, val]: other.params)
        add_param(key, val);

    for (auto const& [key, vals] : other.data) {
        add_data(key);
        for (auto const& val : vals)
            push_data(key, val);
    }
}

DataSlide DataSlide::copy_params(const DataSlide& other) {
    DataSlide slide;
    for (auto const& [key, val]: other.params)
        slide.add_param(key, val);

    return slide;
}


std::vector<std::vector<double>> DataSlide::get_data(const std::string& s) const {
    if (!data.count(s))
        return std::vector<std::vector<double>>();

    size_t N = data.at(s).size();
    if (N == 0)
        return std::vector<std::vector<double>>();
    
    size_t M = data.at(s)[0].size();
    std::vector<std::vector<double>> d(N, std::vector<double>(M));

    for (uint32_t i = 0; i < N; i++) {
        std::vector<Sample> di = data.at(s)[i];
        if (di.size() != M)
            throw std::invalid_argument("Stored data is not square.");

        for (uint32_t j = 0; j < M; j++)
            d[i][j] = di[j].get_mean();
    }

    return d;
}

std::vector<std::vector<double>> DataSlide::get_std(const std::string& s) const {
    if (!data.count(s))
        return std::vector<std::vector<double>>();

    size_t N = data.at(s).size();
    if (N == 0)
        return std::vector<std::vector<double>>();
    
    size_t M = data.at(s)[0].size();
    std::vector<std::vector<double>> d(N, std::vector<double>(M));

    for (uint32_t i = 0; i < N; i++) {
        std::vector<Sample> di = data.at(s)[i];
        if (di.size() != M)
            throw std::invalid_argument("Stored data is not square.");

        for (uint32_t j = 0; j < M; j++)
            d[i][j] = di[j].get_std();
    }

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

    for (auto const &[key, samples_vv] : data) {
        std::vector<std::string> sample_buffer1;
        for (auto const &samples_v : samples_vv) {
            std::vector<std::string> sample_buffer2; 
            for (auto sample : samples_v)
                sample_buffer2.push_back(sample.to_string(record_error));

            sample_buffer1.push_back("[" + join(sample_buffer2, ", ") + "]");
        }

        buffer.push_back("\"" + key + "\": [" + join(sample_buffer1, ", ") + "]");
    }

    for (auto const &[key, samples] : data) {
        size_t N = samples.size();
        std::vector<std::string> sample_buffer1(N);
        for (uint32_t i = 0; i < N; i++) {
            size_t M = samples[i].size();
            std::vector<std::string> sample_buffer2(M);
            for (uint32_t j = 0; j < M; j++)
                sample_buffer2[j] = samples[i][j].to_string(record_error);

            sample_buffer1[i] = "[" + join(sample_buffer2, ", ") + "]";
        }

        buffer.push_back("\"" + key + "\": [" + join(sample_buffer1, ", ") + "]");
    }

    s += join(buffer, delim);
    return s;
}

bool DataSlide::congruent(const DataSlide &ds, const var_t_eq& equality_comparator) {
    if (!params_eq(params, ds.params, equality_comparator))
        return false;

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
        std::string error_message = ss.str();
        throw std::invalid_argument(error_message);
    }

    DataSlide dn(params); 

    for (auto const &[key, samples] : data) {
        dn.add_data(key);
        for (uint32_t i = 0; i < samples.size(); i++) {
            if (samples[i].size() != ds.data.at(key)[i].size()) {
                std::string error_message = "Samples with key '" + key + "' have incongruent length and cannot be combined.";
                throw std::invalid_argument(error_message);
            }

            dn.push_data(key, combine_samples(samples[i], ds.data.at(key)[i]));
        }
    }

    return dn;
}

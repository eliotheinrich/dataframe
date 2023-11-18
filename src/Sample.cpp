#include "Sample.h"

#include <stdexcept>

Sample::Sample(const std::string &s) {
    if (s.front() == '[' && s.back() == ']') {
        std::string trimmed = s.substr(1, s.length() - 2);
        std::vector<uint32_t> pos;
        for (uint32_t i = 0; i < trimmed.length(); i++) {
            if (trimmed[i] == ',')
                pos.push_back(i);
        }

        if (pos.size() != 2)
            throw std::invalid_argument("Invalid string provided to Sample(std::string).");

        mean = std::stof(trimmed.substr(0, pos[0]));
        std = std::stof(trimmed.substr(pos[0]+1, pos[1]));
        num_samples = std::stoi(trimmed.substr(pos[1]+1, trimmed.length()-1));
    } else {
        mean = std::stof(s);
        std = 0.;
        num_samples = 1;
    }
}

Sample Sample::combine(const Sample &other) const {
    uint32_t combined_samples = this->num_samples + other.get_num_samples();
    if (combined_samples == 0) return Sample();
    
    double samples1f = get_num_samples(); double samples2f = other.get_num_samples();
    double combined_samplesf = combined_samples;

    double combined_mean = (samples1f*this->get_mean() + samples2f*other.get_mean())/combined_samplesf;
    double combined_std = std::pow((samples1f*(std::pow(this->get_std(), 2) + std::pow(this->get_mean() - combined_mean, 2))
                                    + samples2f*(std::pow(other.get_std(), 2) + std::pow(other.get_mean() - combined_mean, 2))
                                    )/combined_samplesf, 0.5);

    return Sample(combined_mean, combined_std, combined_samples);
}

Sample Sample::collapse(const std::vector<Sample> &samples) {
    Sample s = samples[0];
    for (uint32_t i = 1; i < samples.size(); i++) {
        s = s.combine(samples[i]);
    }

    return s;
}

std::vector<double> Sample::get_means(const std::vector<Sample> &samples) {
    std::vector<double> v;
    for (auto const &s : samples)
        v.push_back(s.get_mean());
    return v;
}

std::string Sample::to_string(bool full_sample) const {
    if (full_sample) {
        std::string s = "[";
        s += std::to_string(this->mean) + ", " + std::to_string(this->std) + ", " + std::to_string(this->num_samples) + "]";
        return s;
    } else {
        return std::to_string(this->mean);
    }
}
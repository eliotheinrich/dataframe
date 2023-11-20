#pragma once

#include "utils.h"

#include <vector>
#include <string>
#include <math.h>
#include <numeric>

class Sample {
    private:
        double mean;
        double std;
        uint32_t num_samples;

	public:
		Sample() : mean(0.), std(0.), num_samples(0) {}
        Sample(double mean) : mean(mean), std(0.), num_samples(1) {}
		Sample(double mean, double std, uint32_t num_samples) : mean(mean), std(std), num_samples(num_samples) {}

		template<class T>
		Sample(const std::vector<T> &v) {
			num_samples = v.size();
			mean = std::accumulate(v.begin(), v.end(), 0.0);
			double sum = 0.0;
			for (auto const t : v) {
				sum += std::pow(t - mean, 2.0);
			}

			std = std::sqrt(sum/(num_samples - 1.));
		}

		Sample(const std::string &s);

        double get_mean() const {
			return this->mean;
		}
		void set_mean(double mean) {
			this->mean = mean;
		}

        double get_std() const {
			return this->std;
		}
		void set_std(double std) {
			this->std = std;
		}

        uint32_t get_num_samples() const {
			return this->num_samples;
		}
		void set_num_samples(uint32_t num_samples) {
			this->num_samples = num_samples;
		}

        Sample combine(const Sample &other) const;

		static std::vector<double> get_means(const std::vector<Sample> &samples);

		std::string to_string(bool full_sample = false) const;
};

std::vector<Sample> combine_samples(const std::vector<Sample>& samples1, const std::vector<Sample>& samples2);
Sample collapse_samples(const std::vector<Sample>& samples);
std::vector<Sample> collapse_samples(const std::vector<std::vector<Sample>>& samples);

class data_t {
    private:
        std::map<std::string, std::vector<Sample>> data;
    
    public:
        data_t()=default;

        void emplace(const std::string &key, const std::vector<Sample> &samples) {
            data.emplace(key, samples);
        }

        void emplace(const std::string &key, const std::vector<double> &doubles) {
            size_t N = doubles.size();
            std::vector<Sample> samples(N);
            for (uint32_t i = 0; i < N; i++)
                samples[i] = Sample(doubles[i]);

            emplace(key, samples);
        }

        void emplace(const std::string &key, double d) {
            std::vector<Sample> sample{Sample(d)};
			emplace(key, sample);
        }

		void emplace(const std::string &key, Sample s) {
			std::vector<Sample> sample{s};
			emplace(key, sample);
		}

		auto begin() const {
			return data.begin();
		}

		auto end() const {
			return data.end();
		}
		
		int count(const std::string &key) const {
			return data.count(key);
		}

		const std::vector<Sample>& operator[](const std::string& key) const {
        	return data.at(key);
    	}

		std::vector<Sample>& operator[](const std::string& key) {
			return data[key];
		}

};
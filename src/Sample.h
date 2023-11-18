#pragma once

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

		static Sample collapse(const std::vector<Sample> &samples);

		static std::vector<double> get_means(const std::vector<Sample> &samples);

		std::string to_string(bool full_sample = false) const;
};

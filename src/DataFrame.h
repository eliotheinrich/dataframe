#pragma once

#include "DataSlide.h"
#include "utils.h"
#include "types.h"

#include <optional>
#include <set>

class DataFrame {
	private:
		bool qtable_initialized;
		// qtable stores a list of key: {val: corresponding_slide_indices}
		std::map<std::string, std::vector<std::vector<uint32_t>>> qtable;
		std::map<std::string, std::vector<var_t>> key_vals;

		uint32_t corresponding_ind(
			const var_t& v, 
			const std::vector<var_t>& vals, 
			const std::optional<dataframe_utils::var_t_eq>& comp = std::nullopt
		);
		void init_qtable();
		void promote_field(const std::string& s);
		std::set<uint32_t> compatible_inds(const Params& constraints);

	public:
		double atol;
		double rtol;

		Params params;
		Params metadata;
		std::vector<DataSlide> slides;

		friend class ParallelCompute;
		
		DataFrame() : atol(ATOL), rtol(RTOL) {}

		DataFrame(const std::vector<DataSlide>& slides) : atol(ATOL), rtol(RTOL) {
			for (uint32_t i = 0; i < slides.size(); i++) add_slide(slides[i]); 
		}

		DataFrame(const Params& params, const std::vector<DataSlide>& slides) : atol(ATOL), rtol(RTOL) {
			add_param(params);
			for (uint32_t i = 0; i < slides.size(); i++) add_slide(slides[i]); 
		}

		DataFrame(const std::string& s);
		DataFrame(const DataFrame& other);

		void add_slide(const DataSlide& ds);
		bool remove_slide(uint32_t i);

		template <typename T>
		void add_metadata(const std::string& s, const T & t) {
			metadata[s] = t;
		}

		void add_metadata(const Params &params) {
			for (auto const &[key, field] : params)
				add_metadata(key, field);
		}

		template <typename T>
		void add_param(const std::string& s, const T & t) { 
			params[s] = t; 
			qtable_initialized = false;
		}


		void add_param(const Params &params) {
			for (auto const &[key, field] : params)
				add_param(key, field);

			qtable_initialized = false;
		}

		bool contains(const std::string& s) const {
			return params.count(s);
		}

		var_t get(const std::string& s) const {
			if (params.count(s)) return get_param(s);
			else return get_metadata(s);
		}

		var_t get_param(const std::string& s) const {
			return params.at(s);
		}

		var_t get_metadata(const std::string& s) const {
			return metadata.at(s);
		}

		bool remove(const std::string& s) {
			if (params.count(s)) return remove_param(s);
			else return remove_metadata(s);
		}

		bool remove_param(const std::string& s) {
			qtable_initialized = false;
			return params.erase(s);
		}

		bool remove_metadata(const std::string& s) {
			return metadata.erase(s);
		}

		std::string to_string(bool record_error=false) const;
		void write_json(std::string filename, bool record_error=false) const;

		bool field_congruent(std::string s) const;
		void promote_params();

		DataFrame filter(const std::vector<Params>& constraints, bool invert = false);
		std::vector<query_t> query(const std::vector<std::string>& keys, const Params& constraints, bool unique=false, bool error=false);
		void reduce();
		DataFrame combine(const DataFrame &other) const;
};
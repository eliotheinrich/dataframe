#include "DataFrame.h"

#include <fstream>

using namespace dataframe_utils;

uint32_t DataFrame::corresponding_ind(
	const var_t& v, 
	const std::vector<var_t>& vals, 
	const std::optional<var_t_eq>& comp
) {
	var_t_eq equality_comparator = comp.value_or(var_t_eq{atol, rtol});

	for (uint32_t i = 0; i < vals.size(); i++) {
		if (equality_comparator(v, vals[i]))
			return i;
	}

	return -1;
}


void DataFrame::init_qtable() {
	var_t_eq equality_comparator(atol, rtol);
	key_vals = std::map<std::string, std::vector<var_t>>();

	for (auto const &slide : slides) {
		for (auto const &[key, tar_val] : slide.params) {
			if (!key_vals.count(key))
				key_vals[key] = std::vector<var_t>();
			
			auto result = std::find_if(key_vals[key].begin(), key_vals[key].end(), [tar_val, &equality_comparator](const var_t& val) {
				return equality_comparator(tar_val, val);
			});

			if (result == key_vals[key].end())
				key_vals[key].push_back(tar_val);
		}
	}

	// Setting up qtable indices
	for (auto const &[key, vals] : key_vals)
		qtable[key] = std::vector<std::vector<uint32_t>>(vals.size());

	for (uint32_t n = 0; n < slides.size(); n++) {
		auto slide = slides[n];
		for (auto const &[key, vals] : key_vals) {
			var_t val = slide.params[key];
			uint32_t idx = corresponding_ind(val, vals, equality_comparator);
			if (idx == (uint32_t) -1)
				throw std::invalid_argument("Error in init_qtable.");

			qtable[key][idx].push_back(n);
		}
	}

	qtable_initialized = true;
}

void DataFrame::promote_field(const std::string& s) {
	add_param(s, slides.begin()->get_param(s));
	for (auto &slide : slides)
		slide.remove(s);
}

std::set<uint32_t> DataFrame::compatible_inds(const Params& constraints) {
	if (!qtable_initialized)
		init_qtable();

	// Check if any keys correspond to mismatched Frame-level parameters, in which case return nothing
	var_t_eq equality_comparator(atol, rtol);
	for (auto const &[key, val] : constraints) {
		if (params.count(key) && !(equality_comparator(params[key], val)))
			return std::set<uint32_t>();
	}

	// Determine which constraints are relevant, i.e. correspond to existing Slide-level parameters
	Params relevant_constraints;
	for (auto const &[key, val] : constraints) {
		if (!params.count(key))
			relevant_constraints[key] = val;
	}

	std::set<uint32_t> inds;
	for (uint32_t i = 0; i < slides.size(); i++) inds.insert(i);

	for (auto const &[key, val] : relevant_constraints) {
		// Take set intersection
		std::set<uint32_t> tmp;
		uint32_t idx = corresponding_ind(val, key_vals[key], equality_comparator);
		if (idx == (uint32_t) -1)
			continue;

		for (auto const i : qtable[key][idx]) {
			if (inds.count(i))
				tmp.insert(i);
		}

		inds = tmp;
	}
	
	return inds;
}

DataFrame::DataFrame(const std::string& s) {
	nlohmann::json data = nlohmann::json::parse(s);
	for (auto const &[key, val] : data["params"].items())
		params[key] = parse_json_type(val);
	
	if (data.contains("metadata")) {
		for (auto const &[key, val] : data["metadata"].items())
			metadata[key] = parse_json_type(val);
	}

	if (metadata.count("atol"))
		atol = std::get<double>(metadata.at("atol"));
	else
		atol = ATOL;

	if (metadata.count("rtol"))
		rtol = std::get<double>(metadata.at("rtol"));
	else
		rtol = RTOL;

	for (auto const &slide_str : data["slides"])
		add_slide(DataSlide(slide_str.dump()));
}

DataFrame::DataFrame(const DataFrame& other) : atol(other.atol), rtol(other.rtol) {
	for (auto const& [key, val] : other.params)
		params[key] = val;
	
	for (auto const& [key, val] : other.metadata)
		metadata[key] = val;
	
	for (auto const& slide : other.slides)
		add_slide(DataSlide(slide));
}

void DataFrame::add_slide(const DataSlide& ds) {
	slides.push_back(ds);
	qtable_initialized = false;
}

bool DataFrame::remove_slide(uint32_t i) {
	if (i >= slides.size())
		return false;

	slides.erase(slides.begin() + i);
	qtable_initialized = false;
	return true;
}

std::string DataFrame::to_string(bool record_error) const {
	std::string s = "";

	s += "{\n\t\"params\": {\n";

	s += params_to_string(params, 2);

	s += "\n\t},\n\t\"metadata\": {\n";

	s += params_to_string(metadata, 2);

	s += "\n\t},\n\t\"slides\": [\n";

	int num_slides = slides.size();
	std::vector<std::string> buffer;
	for (int i = 0; i < num_slides; i++)
		buffer.push_back("\t\t{\n" + slides[i].to_string(3, true, record_error) + "\n\t\t}");

	s += join(buffer, ",\n");

	s += "\n\t]\n}\n";

	return s;
}

void DataFrame::write_json(std::string filename, bool record_error) const {
	std::string s = to_string(record_error);

	// Save to file
	if (!std::remove(filename.c_str())) std::cout << "Deleting old data\n";

	std::ofstream output_file(filename);
	output_file << s;
	output_file.close();
}

bool DataFrame::field_congruent(std::string s) const {
	if (slides.size() == 0) return true;

	DataSlide first_slide = slides[0];

	if (!first_slide.contains(s)) return false;

	var_t first_slide_val = first_slide.get_param(s);

	var_t_eq equality_comparator(atol, rtol);
	for (auto slide : slides) {
		if (!slide.contains(s)) return false;
		if (!equality_comparator(slide.get_param(s), first_slide_val)) return false;
	}

	return true;
}

void DataFrame::promote_params() {
	var_t_eq equality_comparator(atol, rtol);
	if (slides.size() == 0) return;

	DataSlide first_slide = slides[0];

	std::vector<std::string> keys;
	for (auto const &[key, _] : first_slide.params) keys.push_back(key);
	for (auto key : keys)
		if (field_congruent(key)) promote_field(key);
}

DataFrame DataFrame::filter(const std::vector<Params>& constraints, bool invert) {
	std::set<uint32_t> inds;
	for (auto const &constraint : constraints) {
		auto c_inds = compatible_inds(constraint);
		std::set<uint32_t> ind_union;

		std::set_union(
			inds.begin(), inds.end(),
			c_inds.begin(), c_inds.end(),
			std::inserter(ind_union, ind_union.begin())
		);
		inds = ind_union;
	}

	if (invert) {
		std::vector<uint32_t> all_inds(this->slides.size());
		std::iota(all_inds.begin(), all_inds.end(), 0);

		std::set<uint32_t> all_inds_set(all_inds.begin(), all_inds.end());

		std::set<uint32_t> set_sd;
		std::set_symmetric_difference(
			inds.begin(), inds.end(),
			all_inds_set.begin(), all_inds_set.end(),
			std::inserter(set_sd, set_sd.begin())
		);
		inds = set_sd;
	}
	
	std::vector<DataSlide> slides;
	for (auto i : inds)
		slides.push_back(this->slides[i]);
	
	return DataFrame(params, slides);
}

query_result DataFrame::query(const std::vector<std::string>& keys, const Params& constraints, bool unique, bool error) {
	if (unique) {
		auto result = query(keys, constraints);
		return std::visit(make_query_unique(atol, rtol), result);
	}

	// Determine indices of slides which respect the given constraints
	auto inds = compatible_inds(constraints);
	
	// Constraints yield no valid slides, so return nothing
	if (inds.empty())
		return query_result{std::vector<var_t>()};

	// Compile result of query
	std::vector<query_t> result;
	
	for (auto const& key : keys) {
		query_t key_result;
		if (params.count(key)) {
			key_result = query_t{params[key]};
		} else if (metadata.count(key)) {
			key_result = query_t{metadata[key]};
		} else if (slides[*inds.begin()].params.count(key)) {
			std::vector<var_t> param_vals;
			for (auto const i : inds)
				param_vals.push_back(slides[i].params[key]);
			key_result = query_t{param_vals};
		} else {
			std::vector<std::vector<double>> data_vals;
			if (error) {
				for (auto const i : inds)
					data_vals.push_back(slides[i].get_std(key));
			} else {
				for (auto const i : inds)
					data_vals.push_back(slides[i].get_data(key));
			}
			key_result = query_t{data_vals};
		}

		result.push_back(key_result);
	}

	if (result.size() == 1)
		return query_result{result[0]};
	else
		return query_result{result};

}

void DataFrame::reduce() {
	var_t_eq equality_comparator(atol, rtol);

	std::vector<DataSlide> new_slides;

	std::set<uint32_t> reduced;
	for (uint32_t i = 0; i < slides.size(); i++) {
		if (reduced.count(i))
			continue;

		DataSlide slide(slides[i]);
		auto inds = compatible_inds(slide.params);
		for (auto const j : inds) {
			if (i == j) continue;
			slide = slide.combine(slides[j], equality_comparator);
			reduced.insert(j);
		} 
		new_slides.push_back(slide);
	}

	slides = new_slides;
}

DataFrame DataFrame::combine(const DataFrame &other) const {
	if (params.empty() && slides.empty())
		return DataFrame(other);
	else if (other.params.empty() && other.slides.empty())
		return DataFrame(*this);
	
	std::set<std::string> self_frame_params;
	for (auto const& [k, _] : params)
		self_frame_params.insert(k);

	std::set<std::string> other_frame_params;
	for (auto const& [k, _] : other.params)
		other_frame_params.insert(k);

	// both_frame_params is the intersection of keys of params and other.params
	std::set<std::string> both_frame_params;
	for (auto const& k : self_frame_params) {
		if (other_frame_params.count(k))
			both_frame_params.insert(k);
	}

	// Erase keys which appear in both frame params
	std::set<std::string> to_erase1;
	for (auto const& k : self_frame_params) {
		if (other_frame_params.count(k))
			to_erase1.insert(k);
	}

	std::set<std::string> to_erase2;
	for (auto const& k : other_frame_params) {
		if (self_frame_params.count(k))
			to_erase2.insert(k);
	}

	for (auto const& k : to_erase1)
		self_frame_params.erase(k);

	for (auto const& k : to_erase2)
		other_frame_params.erase(k);

	// self_frame_params and other_frame_params now only contain parameters unique to that frame

	DataFrame df;
	Params self_slide_params;
	Params other_slide_params;

	var_t_eq equality_comparator(atol, rtol);
	for (auto const& k : both_frame_params) {
		if (equality_comparator(params.at(k), other.params.at(k)))
			df.add_param(k, params.at(k));
		else {
			self_slide_params[k] = params.at(k);
			other_slide_params[k] = other.params.at(k);
		}
	}

	for (auto const& k : self_frame_params)
		self_slide_params[k] = params.at(k);
	
	for (auto const& k : other_frame_params)
		other_slide_params[k] = other.params.at(k);

	for (auto const& slide : slides) {
		DataSlide ds(slide);
		for (auto const& [k, v] : self_slide_params)
			ds.add_param(k, v);
		
		df.add_slide(ds);
	}

	for (auto const& slide : other.slides) {
		DataSlide ds(slide);
		for (auto const& [k, v] : other_slide_params)
			ds.add_param(k, v);
		
		df.add_slide(ds);
	}

	df.promote_params();
	df.reduce();

	return df;
}
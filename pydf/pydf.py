import json
import numpy as np

class DataSlide:
	def __init__(self, params, data):
		self.params = params
		self.data = data

	def get(self, key):
		if key in self.params:
			return self.params[key]
		else:
			return self.data[key][:,0]
	
	def get_err(self, key):
		return self.data[key][:,1]

	def get_nruns(self, key):
		return self.data[key][:,2]

def _remove_flat_axes(A):
	shape = np.array(A.shape)
	nonflat_axes = np.where(shape != 1)

	return A.reshape(shape[nonflat_axes])


class DataFrame:
	def __init__(self):
		self.params = {}
		self.slides = []
	
	def __add__(self, other):
		new = DataFrame()
		new.params = {**self.params, **other.params}

		for slide in self.slides:
			new.add_dataslide(slide)
		for slide in other.slides:
			new.add_dataslide(slide)

		return new

	def add_dataslide(self, slide):
		self.slides.append(slide)

	def get_property_with_id(self, key, id):
		return self.slides[id][key]

	def get_param(self, key):
		return self.params[key]

	def get(self, key):
		if key in self.params:
			return self.params[key]
		
		else:
			vals = []
			for slide in self.slides:
				vals.append(slide.get(key))

		vals = np.array(vals)
		return _remove_flat_axes(vals)
	
	def get_err(self, key):
		vals = []
		for slide in self.slides:
			vals.append(slide.get_err(key))
		
		vals = np.array(vals)
		return _remove_flat_axes(vals)

	def get_nruns(self, key):
		vals = []
		for slide in self.slides:
			vals.append(slide.get_nruns(key))
		
		vals = np.array(vals)
		return _remove_flat_axes(vals)
	
	def filter(self, key, val, invert=False):
		if key in self.params:
			if val == self.params[key]:
				return self
			else:
				return DataFrame()
				
		new_df = DataFrame()
		new_df.params = self.params.copy()
		for slide in self.slides:
			if isinstance(val, str):
				keep = val == slide.get(key)
			else:
				keep = np.isclose(slide.get(key), val)

			if invert:
				keep = not keep
			if keep:
				new_df.add_dataslide(slide)
		
		return new_df
	
	def get_filtered(self, key, filter_key, val):
		filtered_df = self.filter(filter_key, val)
		return filtered_df.get(key)

	def get_err_filtered(self, key, filter_key, val):
		filtered_df = self.filter(filter_key, val)
		return filtered_df.get_err(key)

	def get_nruns_filtered(self, key, filter_key, val):
		filtered_df = self.filter(filter_key, val)
		return filtered_df.get_nruns(key)

def parse_datafield(s):
	if list(s.keys())[0] == 'Data':
		sample = s[list(s.keys())[0]]
		return np.array(sample)
	else:
		return np.array([s[list(s.keys())[0]]])

def load_data(filename):
	dataframe = DataFrame()
	with open(filename, 'r') as f:
		json_contents = json.load(f)
		for param_key, param in json_contents['params'].items():
			try:
				dataframe.params = parse_datafield(param)[0]
			except:
				dataframe.params[param_key] = param
		for slide in json_contents['slides']:
			try:
				slide_dict = slide['data']
				vals = {key: parse_datafield(slide_dict[key]) for key in slide_dict.keys()}
			except KeyError:
				slide_dict = slide
				vals = {key: slide_dict[key] for key in slide_dict.keys()}

			params = {}
			data = {}
			for key in slide_dict.keys():
				if isinstance(vals[key], list):
					data[key] = np.array(vals[key])
				else:
					params[key] = vals[key] 
				
			dataframe.add_dataslide(DataSlide(params, data))
	
	return dataframe

def sort_data_by(sorter, *args):
	idxs = np.argsort(sorter)
	return (sorter[idxs], *[arg[idxs] for arg in args])

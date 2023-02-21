import json
import numpy as np

class DataSlide:
	def __init__(self, keys, vals):
		self.data = dict(zip(keys, vals))

	def get(self, key):
		if self.data[key].ndim == 1:
			return self.data[key][0]
		else:
			return self.data[key][:,0]
	
	def get_err(self, key):
		if self.data[key].ndim == 1:
			raise KeyError("Data not found")
		else:
			return self.data[key][:,1]

	def get_nruns(self, key):
		if self.data[key].ndim == 1:
			raise KeyError("Data not found")
		else:
			return self.data[key][:,2]

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
		val = []
		for slide in self.slides:
			val.append(slide.get(key))
		return np.array(val)
	
	def get_err(self, key):
		val = []
		for slide in self.slides:
			val.append(slide.get_err(key))
		return np.array(val)

	def get_nruns(self, key):
		val = []
		for slide in self.slides:
			val.append(slide.get_nruns(key))
		return np.array(val)
	
	def filter(self, key, val):
		new_df = DataFrame()
		new_df.params = self.params.copy()
		for slide in self.slides:
			if np.isclose(slide.get(key), val):
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
	data = DataFrame()
	with open(filename, 'r') as f:
		json_contents = json.load(f)
		for param_key in json_contents['params']:
			data.params[param_key] = parse_datafield(json_contents['params'][param_key])[0]
		for slide in json_contents['slides']:
			keys = list(slide['data'].keys())
			vals = [parse_datafield(slide['data'][key]) for key in keys]
				
			data.add_dataslide(DataSlide(keys, vals))
	
	return data

def sort_data_by(sorter, *args):
	idxs = np.argsort(sorter)
	return (sorter[idxs], *[arg[idxs] for arg in args])


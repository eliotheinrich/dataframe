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

		return np.array(vals).flatten()
	
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

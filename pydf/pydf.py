import json
import numpy as np

def param_equal(s1, s2):
	if isinstance(s1, str) and isinstance(s2, str):
		return s1 == s2
	else:
		return np.isclose(s1, s2)
		
def constraints_stronger(c1, c2):
	for key,val in c1.items():
		if not (key in c2 and param_equal(val, c2[key])):
			return False
	
	return True

class DataSlide:
	def __init__(self, params, data):
		self.params = params
		self.data = data

	def contains(self, key):
		return key in self.params or key in self.data

	def _get(self, key):
		if key in self.params:
			return self.params[key]
		else:
			return self.data[key][:,0]
	
	def _get_err(self, key):
		return self.data[key][:,1]

	def _get_nruns(self, key):
		return self.data[key][:,2]

	def compatible(self, constraints):
		for key, val in constraints.items():
			if not self.contains(key):
				return False

			if not param_equal(val, self.params[key]):
				return False
		
		return True
			

def _remove_flat_axes(A):
	shape = np.array(A.shape)
	if len(shape) == 1 and shape[0] == 1:
		return A

	nonflat_axes = np.where(shape != 1)

	return A.reshape(shape[nonflat_axes])

def _add(list, val):
	for i in list:
		if param_equal(i, val):
			return
		
	list.append(val)
class DataFrame:
	def head(self):
		if "description" in self.params:
			return self.params["description"]
		else:
			return "No description."
		
	def __init__(self):
		self._qtable_initialized = False
		self._qtable = {}

		self.params = {}
		self.slides = []
	
	def _init_qtable(self):
		key_vals = {}
		for slide in self.slides:
			for key, val in slide.params.items():
				if key not in key_vals:
					key_vals[key] = []

				_add(key_vals[key], val)
		
		for key, vals in key_vals.items():
			self._qtable[key] = {}
			for v in vals:
				self._qtable[key][v] = set()

		for n,slide in enumerate(self.slides):
			for key, val in slide.params.items():
				self._qtable[key][val].add(n)

		for k,v in self.params.items():
			self._qtable[k] = v

		self._qtable_initialized = True

	def __len__(self):
		return len(self.slides)
	
	def __add__(self, other):
		new = DataFrame()
		new.params = {**self.params, **other.params}

		for slide in self.slides:
			new.add_dataslide(slide)
		for slide in other.slides:
			new.add_dataslide(slide)

		return new
	
	def query(self, keys, constraints={}, query_error=False):
		if not self._qtable_initialized:
			self._init_qtable()

		# First check if passed key is a Frame level parameter and return if so
		if isinstance(keys, str):
			if keys in self.params:
				return self.params[keys]
			else:
				keys = [keys]
		
		for k,v in constraints.items():
			# stored Frame level param is not equal to param specified by constraint, so return nothing
			if k in self.params and not param_equal(v, self.params[k]):
				return np.array([])

		relevant_constraints = {k: v for k,v in constraints.items() if k not in self.params}

		if relevant_constraints == {}:
			ind = range(0, len(self))
		else:
			ind = set.intersection(*[self._qtable[k][v] for (k,v) in relevant_constraints.items()])

		vals = {key: [] for key in keys}
		for i in ind:
			for key in keys:
				if query_error:
					vals[key].append(self.slides[i]._get_err(key))
				else:
					vals[key].append(self.slides[i]._get(key))

		if len(keys) == 1:
			key = keys[0]

			v = _remove_flat_axes(np.array(vals[key]))
			inds = np.argsort(v)
			return _remove_flat_axes(np.array(vals[key]))


		for key in keys:
			vals[key] = np.array(vals[key])
		
		key = keys[0]
		if vals[key].ndim == 1 and not isinstance(vals[key][0], str):
			inds = np.argsort(vals[key])
			return [_remove_flat_axes(vals[k][inds]) for k in keys]
		else:
			return [_remove_flat_axes(vals[k]) for k in keys]

	
	def query_unique(self, keys): # TODO allow constraints
		if not self._qtable_initialized:
			self._init_qtable()
			
		if isinstance(keys, str):
			keys = [keys]
		
		v = _remove_flat_axes(np.array([list(self._qtable[key].keys()) for key in keys]))
		return v

	def param_congruent(self, key):
		if not self.slides[0].contains(key):
			return False

		val = self.slides[0]._get(key)
		for slide in self.slides:
			if not slide.contains(key):
				return False

			if not param_equal(val, slide._get(key)):
				return False
		
		return True

	def promote(self, key):
		val = self.slides[0]._get(key)
		self.params[key] = val
		for slide in self.slides:
			del slide.params[key]

	def promote_params(self):
		keys = list(self.slides[0].params.keys())
		for key in keys:
			if self.param_congruent(key):
				self.promote(key)

	def add_dataslide(self, slide):
		self._qtable_initialized = False
		self.slides.append(slide)

	def _get(self, key):
		if key in self.params:
			return self.params[key]
		
		else:
			vals = []
			for slide in self.slides:
				vals.append(slide._get(key))

		return _remove_flat_axes(np.array(vals))
	
	def _get_err(self, key):
		vals = []
		for slide in self.slides:
			vals.append(slide._get_err(key))
		
		return _remove_flat_axes(np.array(vals))

	def _get_nruns(self, key):
		vals = []
		for slide in self.slides:
			vals.append(slide._get_nruns(key))
		
		return _remove_flat_axes(np.array(vals))
	
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
				keep = val == slide._get(key)
			else:
				keep = np.isclose(slide._get(key), val)

			if invert:
				keep = not keep
			if keep:
				new_df.add_dataslide(slide)
		
		return new_df
	
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


if __name__ == "__main__":
	data = load_data('test.json')
	print(data.query('num_runs'))
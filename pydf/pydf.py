import json
import numpy as np

def param_equal(s1, s2):
	if isinstance(s1, str) and isinstance(s2, str):
		return s1 == s2
	else:
		return np.isclose(s1, s2)

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


class DataSlide:
	def __init__(self, params = None, data = None):
		self.params = params or {}
		self.data = {key: np.array(vals, dtype=float) for key, vals in data.items()} if data is not None else {}
	
	def add_param(self, key: str, val):
		self.params[key] = float(val)
	
	def add_data(self, key: str):
		self.data[key] = np.array([], dtype=float)
  
	def push_data(self, key: str, val):
		self.data[key] = np.append(self.data[key], float(val))

	def contains(self, key: str):
		return key in self.params or key in self.data

	def _get(self, key: str):
		if key in self.params:
			return self.params[key]
		else:
			return self.data[key]

	def compatible(self, constraints):
		for key, val in constraints.items():
			if not self.contains(key):
				return False

			if not param_equal(val, self.params[key]):
				return False
		
		return True	

	def as_dict(self):
		return {**self.params, **{key: list(vals.astype(float)) for key, vals in self.data.items()}}

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
  
		self_df_params = set(self.params.keys())
		other_df_params = set(other.params.keys())

		self_ds_params = {}
		other_ds_params = {}
  
		for key in self_df_params.intersection(other_df_params):
			if param_equal(self.params[key], other.params[key]):
				new.add_param(key, self.params[key])
			else:
				self_ds_params[key] = self.params[key]
				other_ds_params[key] = other.params[key]
		

		for slide in self.slides:
			for key, val in self_ds_params.items():
				slide.add_param(key, val)
			new.add_slide(slide)
				
		for slide in other.slides:
			for key, val in other_ds_params.items():
				slide.add_param(key, val)
			new.add_slide(slide)

		return new
	
	def query(self, keys, constraints={}):
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

		vals = {key: [self.slides[i]._get(key) for i in ind] for key in keys}

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

	
	def query_unique(self, keys, constraints = {}):
		query_result = self.query(keys, constraints)
		if not isinstance(query_result, np.ndarray):
			query_result = [query_result]

		return sorted(list(set(query_result)))

	def param_congruent(self, key: str):
		if not self.slides[0].contains(key):
			return False

		val = self.slides[0]._get(key)
		for slide in self.slides:
			if not slide.contains(key):
				return False

			if not param_equal(val, slide._get(key)):
				return False
		
		return True

	def promote(self, key: str):
		val = self.slides[0]._get(key)
		self.params[key] = val
		for slide in self.slides:
			del slide.params[key]

	def promote_params(self):
		keys = list(self.slides[0].params.keys())
		for key in keys:
			if self.param_congruent(key):
				self.promote(key)

	def add_slide(self, slide: DataSlide):
		self._qtable_initialized = False
		self.slides.append(slide)

	def add_param(self, key: str, val):
		self._qtable_initialized = False
		self.params[key] = val
	

	def _get(self, key: str):
		if key in self.params:
			return self.params[key]
		
		else:
			vals = []
			for slide in self.slides:
				vals.append(slide._get(key))

		return _remove_flat_axes(np.array(vals))
	
	def filter(self, key: str, val, invert=False):
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
				new_df.add_slide(slide)
		
		return new_df

	def __str__(self):
		fields = {'params': self.params, 'slides': self.slides}
		return json.dumps(fields, 
                    	  default=lambda o: o.as_dict(),
                          allow_nan=False, 
                          indent=4
                        )

	def write_json(self, filename: str):
		fields = {'params': self.params, 'slides': self.slides}

		with open(filename, 'w') as f:
			json.dump(fields, f, 
             		  default=lambda o: o.as_dict(), 
                 	  allow_nan=False, indent=4
                    )


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
			try: # For backcompatibility with rs version
				dataframe.params = parse_datafield(param)[0]
			except:
				dataframe.params[param_key] = param
		for slide in json_contents['slides']:
			try: # For backcompatibility with rs version
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
				
			dataframe.add_slide(DataSlide(params, data))
	
	return dataframe

def sort_data_by(sorter, *args):
	idxs = np.argsort(sorter)
	return (sorter[idxs], *[arg[idxs] for arg in args])



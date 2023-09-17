from typing import Any
import dataframe.dataframe_bindings as _df

class DataSlide:
    def __init__(self, data: str | dict | None = None):
        if data is None:
            self._dataslide = _df.DataSlide()
        else:
            self._dataslide = _df.DataSlide(data)

    @property
    def params(self):
        return self._dataslide.params
    
    def add_param(self, s: str, p: int | float | str):
        self._dataslide.add_param(s, p)
    
    def add_data(self, s: str):
        self._dataslide.add_data(s)

    def push_data(self, s: str, p: float):
        self._dataslide.push_data(s, p)
    
    def remove(self, s: str):
        self._dataslide.remove(s)
    
    def __contains__(self, s: str) -> bool:
        return s in self._dataslide
    
    def __getitem__(self, s: str) -> int | float | str:
        return self._dataslide[s]

    def __setitem__(self, s: str, val: int | float | str):
        self._dataslide.__setitem__(s, val)

    def __str__(self) -> str:
        return str(self._dataslide)

    def congruent(self, slide) -> bool:
        return self._dataslide.congruent(slide)

    def combine(self, slide):
        return self._dataslide.combine(slide)


class _SlideContainer:
    def __init__(self, slides):
        self._slides = slides
    
    def __getitem__(self, i):
        return DataSlide(self._slides[i])
    
    def __len__(self):
        return len(self._slides)

class DataFrame:
    def __init__(self, data: list | str | _df.DataFrame | None = None, params = None):
        if params is not None:
            self._dataframe = _df.DataFrame(params, data) 
        elif isinstance(data, DataFrame):
            self._dataframe = data._dataframe
        elif data is None:
            self._dataframe = _df.DataFrame()
        else:
            self._dataframe = _df.DataFrame(data)
        
        self._slides = _SlideContainer(self._dataframe.slides)
        
    @property
    def params(self):
        return self._dataframe.params
    
    @property
    def slides(self):
        return self._slides
        
    def add_slide(self, slide):
        self._dataframe.add_slide(slide._dataslide)
    
    def add_param(self, s: str, p: int | float | str):
        self._dataframe.add_param(s, p)
    
    def add_metadata(self, s: str, p: int | float | str):
        self._dataframe.add_metadata(s, p)
    
    def remove(self, s: str):
        self._dataframe.remove(s)
    
    def __contains__(self, s: str) -> bool:
        return s in self._dataframe
    
    def __setitem__(self, s: str, val: int | float | str):
        self._dataframe.__setitem__(s, val)
    
    def __getitem__(self, s: str) -> int | float | str:
        return self._dataframe[s]

    def __str__(self) -> str:
        return str(self._dataframe)
    
    def __add__(self, other):
        new = DataFrame() 
        new._dataframe = self._dataframe + other._dataframe
        return new

    def write_json(self, filename: str):
        self._dataframe.write_json(filename)
    
    def promote_params(self):
        self._dataframe.promote_params()
    
    def filter(self, constraints: dict, invert: bool = False):
        return DataFrame(self._dataframe.filter(constraints, invert))
    
    def modify_slides(self, func):
        for slide in self._dataframe.slides:
            func(slide)

        self._dataframe = _df.DataFrame(self.params, self._dataframe.slides) 
    
    def query(self, keys: list | str, constraints: dict | None = None, unique: bool = False) -> list:
        if isinstance(keys, str):
            keys = [keys]
        if constraints is None:
            constraints = {}
        return self._dataframe.query(keys, constraints, unique)
          
    def query_unique(self, keys: list | str, constraints: dict | None = None) -> list:
        return self.query(keys, constraints, unique=True)

class ParallelCompute:
    def __init__(self, configs, num_threads: int = 1):
        self._pc = _df.ParallelCompute(configs, num_threads)
    
    def set_serialize(self, serialize: bool):
        self._pc = serialize
    
    def get_serialize(self) -> bool:
        return self._pc.serialize
    
    @property
    def dataframe(self):
        return self._pc.dataframe
    
    @property
    def serialize(self):
        return self._pc.serialize
    
    def compute(self, verbose: bool = False):
        self._pc.compute(verbose)
    
    def write_json(self, filename: str):
        self._pc.write_json(filename)

    def write_serialize_json(self, filename: str):
        self._pc.write_serialize_json(filename)

def load_data(filename: str) -> DataFrame:
    with open(filename, 'r') as f:
        s = f.read()
    
    return DataFrame(s)

def load_json(filename: str, verbose: bool = False) -> list:
    return _df.load_json(filename, verbose)

def write_config(params: list) -> str:
    return _df.write_config(params)
    

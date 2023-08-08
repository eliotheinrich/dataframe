import dataframe.dataframe_bindings as _df

def load_json(filename: str, verbose: bool = False) -> list:
    return _df.load_json(filename, verbose)

def write_config(params: list) -> str:
    return _df.write_config(params)
    

class DataSlide:
    def __init__(self):
        self._dataslide = _df.DataSlide()
    
    def __init__(self, data: str | dict):
        self._dataslide = _df.DataSlide(data)
    
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

    def __str__(self) -> str:
        return str(self._dataslide)

    def congruent(self, slide) -> bool:
        return self._dataslide.congruent(slide)

    def combine(self, slide):
        return self._dataslide.combine(slide)

class DataFrame:
    def __init__(self):
        self._dataframe = _df.DataFrame()
        self.params = self._dataframe.params
    
    def __init__(self, data):
        self._dataframe = _df.DataFrame(data)
    
    def add_slide(self, slide):
        self._dataframe.add_slide(slide)
    
    def add_param(self, s: str, p: int | float | str):
        self._dataframe.add_param(s, p)
    
    def remove(self, s: str):
        self._dataframe.remove(s)
    
    def __contains__(self, s: str) -> bool:
        return s in self._dataframe
    
    def __getitem__(self, s: str) -> int | float | str:
        return self._dataframe[s]

    def __str__(self) -> str:
        return str(self._dataframe)

    def write_json(self, filename: str):
        self._dataframe.write_json(filename)
    
    def promote_params(self):
        self._dataframe.promote_params()
    
    def query(self, keys: list, constraints: dict, unique: bool = False) -> list:
        return self._dataframe.query(keys, constraints, unique)
          

class ParallelCompute:
    def __init__(self, configs):
        self._pc = _df.ParallelCompute(configs)
        self.dataframe = None 
    
    def compute(self, verbose: bool = False):
        self._pc.compute(verbose)
    
    def write_json(self, filename: str):
        self._pc.write_json(filename)
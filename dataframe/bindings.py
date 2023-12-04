from typing import Any
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from dataframe.dataframe_bindings import *


import time

#class DataSlide:
#    def __init__(self, data: str | dict | None = None):
#        if data is None:
#            self._dataslide = _df.DataSlide()
#        else:
#            self._dataslide = _df.DataSlide(data)
#
#    @property
#    def params(self):
#        return self._dataslide.params
#    
#    def add_param(self, s: str, p: int | float | str):
#        self._dataslide.add_param(s, p)
#    
#    def add_data(self, s: str):
#        self._dataslide.add_data(s)
#
#    def push_data(self, s: str, p: float):
#        self._dataslide.push_data(s, p)
#    
#    def remove(self, s: str):
#        self._dataslide.remove(s)
#    
#    def __contains__(self, s: str) -> bool:
#        return s in self._dataslide
#    
#    def __getitem__(self, s: str) -> int | float | str:
#        return self._dataslide[s]
#
#    def __setitem__(self, s: str, val: int | float | str):
#        self._dataslide.__setitem__(s, val)
#
#    def __str__(self) -> str:
#        return str(self._dataslide)
#
#    def congruent(self, slide) -> bool:
#        return self._dataslide.congruent(slide)
#
#    def combine(self, slide):
#        return self._dataslide.combine(slide)
#
#
#class _SlideContainer:
#    def __init__(self, slides):
#        self._slides = slides
#    
#    def __getitem__(self, i):
#        return DataSlide(self._slides[i])
#    
#    def __len__(self):
#        return len(self._slides)
#
#class DataFrame:
#    def __init__(self, data: list | str | _df.DataFrame | None = None, params = None):
#        if params is not None:
#            self._dataframe = _df.DataFrame(params, data) 
#        elif isinstance(data, DataFrame):
#            self._dataframe = data._dataframe
#        elif data is None:
#            self._dataframe = _df.DataFrame()
#        else:
#            self._dataframe = _df.DataFrame(data)
#        
#        self._slides = _SlideContainer(self._dataframe.slides)
#        
#    @property
#    def params(self):
#        return self._dataframe.params
#     
#    @property
#    def slides(self):
#        return self._slides
#
#    @property
#    def atol(self):
#        return self._dataframe.atol
#
#    @atol.setter
#    def atol(self, atol: float):
#        self._dataframe.atol = atol
#
#    @property
#    def rtol(self):
#        return self._dataframe.rtol
#
#    @rtol.setter
#    def rtol(self, rtol: float):
#        self._dataframe.rtol = rtol
#        
#    def add_slide(self, slide):
#        self._dataframe.add_slide(slide._dataslide)
#    
#    def add_param(self, s: str, p: int | float | str):
#        self._dataframe.add_param(s, p)
#    
#    def add_metadata(self, s: str, p: int | float | str):
#        self._dataframe.add_metadata(s, p)
#    
#    def remove(self, s: str):
#        self._dataframe.remove(s)
#    
#    def __contains__(self, s: str) -> bool:
#        return s in self._dataframe
#    
#    def __setitem__(self, s: str, val: int | float | str):
#        self._dataframe.__setitem__(s, val)
#    
#    def __getitem__(self, s: str) -> int | float | str:
#        return self._dataframe[s]
#
#    def __str__(self) -> str:
#        return str(self._dataframe)
#    
#    def __add__(self, other):
#        new = DataFrame() 
#        new._dataframe = self._dataframe + other._dataframe
#        return new
#
#    def write_json(self, filename: str, record_error: bool = False):
#        self._dataframe.write_json(filename, record_error)
#    
#    def promote_params(self):
#        self._dataframe.promote_params()
#    
#    def filter(self, constraints: dict, invert: bool = False):
#        return DataFrame(self._dataframe.filter(constraints, invert))
#    
#    def modify_slides(self, func):
#        for slide in self._dataframe.slides:
#            func(slide)
#
#        self._dataframe = _df.DataFrame(self.params, self._dataframe.slides) 
#    
#    def query(self, keys: list | str, constraints: dict | None = None, unique: bool = False, error: bool = False) -> list:
#        if isinstance(keys, str):
#            keys = [keys]
#        if constraints is None:
#            constraints = {}
#        return self._dataframe.query(keys, constraints, unique, error)
#          
#    def query_unique(self, keys: list | str, constraints: dict | None = None) -> list:
#        return self.query(keys, constraints, unique=True)


ATOL = 1e-6
RTOL = 1e-5

from abc import ABC, abstractmethod

class Config(ABC):
    def __init__(self, params):
        self.params = params
        self.num_runs = params.setdefault("num_runs", 1)
    
    def __getstate__(self):
        return self.params,
    
    def __setstate__(self, state):
        self.__init__(*state)
    
    def get_nruns(self):
        return self.params["num_runs"]
    
    @abstractmethod
    def compute(self, num_threads):
        pass

class TimeConfig(Config):
    def __init__(self, params, simulator_generator):
        super().__init__(params)

        self.simulator_generator = simulator_generator
        self.simulator_driver = simulator_generator(params)

    def __getstate__(self):
        return self.params, self.simulator_generator
     
    def compute(self, num_threads):
        return self.simulator_driver.generate_dataslide(num_threads)
    
    def clone(self):
        return TimeConfig(self.params, self.simulator_generator)

class FuncConfig(Config):
    def __init__(self, params, function):
        super().__init__(params)
        self.function = function
    
    def compute(self, num_threads):
        return self.function(self.params, num_threads)

    def __getstate__(self):
        return self.params, self.function
    
    def clone(self):
        return FuncConfig(self.params, self.function)

     
class ParallelCompute:
    SERIAL = 0
    POOL = 1
    
    def __init__(self, configs, **metadata):
        self.configs = configs
        
        self.num_threads = metadata.setdefault("num_threads", 1)
        self.num_threads_per_task = metadata.setdefault("num_threads_per_task", 1)
        self.atol = metadata.setdefault("atol", ATOL)
        self.rtol = metadata.setdefault("rtol", RTOL)
        self.average_congruent_runs = metadata.setdefault("average_congruent_runs", True)
        self.parallelization_type = metadata.setdefault("parallelization_type", self.SERIAL)
        self.record_error = metadata.setdefault("record_error", True)
        self.dataframe = DataFrame(self.atol, self.rtol)
        
        self._prev_percent_finished = 0

    def write_json(self, filename):
        s = str(self.dataframe)
        with open(filename, 'w') as f:
            f.write(s)

    def compute(self, verbose=True):
        start_time = time.time()
        
        num_configs = len(self.configs)
        total_configs = []
        for config in self.configs:
            config.clone()
            nruns = config.get_nruns()
            
            for _ in range(nruns):
                total_configs.append(config.clone())
            
        if self.parallelization_type == self.SERIAL:
            results = self.compute_serial(total_configs, verbose)
        elif self.parallelization_type == self.POOL:
            results = self.compute_pool(total_configs, verbose)

        if verbose:
            print("\n", end="")
            
        idx = 0
        for i in range(num_configs):
            slide = results[idx]
            nruns = self.configs[i].get_nruns()
            for _ in range(1, nruns):
                idx += 1
                slide_tmp = results[idx]
                if self.average_congruent_runs:
                    slide = slide.combine(slide_tmp) # TODO custom atol/rtol
                else:
                    self.dataframe.add_slide(slide_tmp)
            
            idx += 1
            self.dataframe.add_slide(slide)
        
        stop_time = time.time()
        duration = stop_time - start_time
        num_jobs = len(total_configs)

        self.dataframe.add_metadata("num_threads", self.num_threads)
        self.dataframe.add_metadata("num_jobs", num_jobs)
        self.dataframe.add_metadata("total_time", duration)
        
        self.dataframe.promote_params()
        if self.average_congruent_runs:
            self.dataframe.reduce()
        
        if verbose:
            print(f"Total runtime: {duration}")
    
    @staticmethod
    def _do_run(config, num_threads):
        slide = config.compute(num_threads)
        slide.add_param(config.params)
        
        return slide
        
    
    def compute_serial(self, total_configs, verbose):
        if verbose:
            print("Computing in serial.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")
            
        run_start = time.time()
        results = []
        for i,config in enumerate(total_configs):
            results.append(ParallelCompute._do_run(config, self.num_threads_per_task))
            
            if verbose:
                self._print_progress(i, len(total_configs), run_start)

        if verbose:
            self._print_progress(len(total_configs), len(total_configs), run_start)
        
        return results
    
    def compute_pool(self, total_configs, verbose):
        if verbose:
            print(f"Computing in parallel. {self.num_threads} threads available.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")
            
        run_start = time.time()

        with Pool(self.num_threads) as pool:
            results = pool.map(partial(ParallelCompute._do_run, num_threads=self.num_threads_per_task), total_configs)

            i = 0
            for _ in results:
                i += 1
                self._print_progress(i, len(total_configs), run_start)
            

        if verbose:
            self._print_progress(len(total_configs), len(total_configs), run_start)
        
        return results
    
    def _print_progress(self, i, N, run_start=None):
        percent_finished = float(i)/N * 100
        ipercent_finished = round(percent_finished)
        if (ipercent_finished != self._prev_percent_finished):
            self._prev_percent_finished = ipercent_finished
            if run_start is not None:
                now = time.time()
                duration = now - run_start
            
            seconds_per_job = duration/i
            remaining_time = int(seconds_per_job * (N - i))
            progress = percent_finished / 100.0
            
            bar_width = 70
            pos = int(bar_width*progress)

            bar = "[" + "="*pos + ">" + " "*(bar_width - pos - 1) + "]"

            num_seconds = remaining_time % 60
            num_minutes = remaining_time // 60
            num_hours = num_minutes // 60
            
            print(f"{bar} [ ETA: {num_hours:02}:{num_minutes:02}:{num_seconds:02} ] {progress*100:.2f} % ", end="\r")


def load_data(filename: str) -> DataFrame:
    with open(filename, 'r') as f:
        s = f.read()
    
    return DataFrame(s)

def load_json(filename: str, verbose: bool = False) -> list:
    return parse_config(filename, verbose)

def write_config(params: list) -> str:
    return paramset_to_string(params)
    

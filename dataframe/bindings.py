from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import concurrent

import time
import tqdm

from dataframe.dataframe_bindings import *


ATOL = 1e-6
RTOL = 1e-5

def parse_config(config):
    for key in config:
        pass


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
        slide = self.simulator_driver.generate_dataslide(num_threads)
        self.params = self.simulator_driver.params
        return slide

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
        self._metadata = metadata

        self.num_threads = metadata.setdefault("num_threads", 1)
        self.num_threads_per_task = metadata.setdefault("num_threads_per_task", 1)
        self.atol = metadata.setdefault("atol", ATOL)
        self.rtol = metadata.setdefault("rtol", RTOL)
        self.average_congruent_runs = metadata.setdefault("average_congruent_runs", True)
        self.parallelization_type = metadata.setdefault("parallelization_type", self.SERIAL)
        self.record_error = metadata.setdefault("record_error", True)
        self.dataframe = DataFrame(self.atol, self.rtol)

        self.num_slides = None

    def write_json(self, filename):
        s = str(self.dataframe)
        with open(filename, 'w') as f:
            f.write(s)

    def compute(self, verbose=True):
        start_time = time.time()

        total_configs = []
        j = 0
        for i, config in enumerate(self.configs):
            config.clone()
            nruns = config.get_nruns()

            for _ in range(nruns):
                id = i if self.average_congruent_runs else j
                total_configs.append((id, config.clone()))
                j += 1

        self.num_slides = len(self.configs) if self.average_congruent_runs else len(total_configs)

        if self.parallelization_type == self.SERIAL:
            results = self.compute_serial(total_configs, verbose)
        elif self.parallelization_type == self.POOL:
            results = self.compute_pool(total_configs, verbose)

        if verbose:
            print("\n", end="")

        for slide in results:
            self.dataframe.add_slide(slide)

        stop_time = time.time()
        duration = stop_time - start_time
        num_jobs = len(total_configs)

        self.dataframe.add_metadata("num_threads", self.num_threads)
        self.dataframe.add_metadata("num_jobs", num_jobs)
        self.dataframe.add_metadata("total_time", duration)
        self.dataframe.add_metadata(self._metadata)

        self.dataframe.promote_params()
        if self.average_congruent_runs:
            self.dataframe.reduce()

        if verbose:
            print(f"Total runtime: {duration:0.0f}")

        return self.dataframe

    @staticmethod
    def _do_run(config, num_threads, id):
        slide = config.compute(num_threads)
        slide.add_param(config.params)

        return id, slide

    def compute_serial(self, total_configs, verbose):
        if verbose:
            print(f"Computing in serial.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")

        slides = [None for _ in range(self.num_slides)]
        if verbose:
            total_configs = tqdm.tqdm(total_configs)
        for i, config in total_configs:
            id, slide = ParallelCompute._do_run(config, self.num_threads_per_task, i)

            slides[id] = slide if slides[id] is None else slides[id].combine(slide, self.atol, self.rtol)

        return slides

    def compute_pool(self, total_configs, verbose):
        if verbose:
            print(f"Computing in parallel. {self.num_threads} threads available.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")

        slides = [None for _ in range(self.num_slides)]
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            futures = [pool.submit(ParallelCompute._do_run, config, self.num_threads_per_task, i) for i,config in total_configs]
            completed_futures = concurrent.futures.as_completed(futures)
            if verbose:
                completed_futures = tqdm.tqdm(completed_futures, total=len(futures))

            for future in completed_futures:
                id, slide = future.result()

                slides[id] = slide if slides[id] is None else slides[id].combine(slide, self.atol, self.rtol)

        return slides

def load_data(filename: str) -> DataFrame:
    with open(filename, 'r') as f:
        s = f.read()

    return DataFrame(s)

def load_json(filename: str, verbose: bool = False) -> list:
    return parse_config(filename, verbose)


def write_config(params: list) -> str:
    return paramset_to_string(params)

#def field_to_string(field) -> str:
#    if isinstance(field, str):
#        return f'"{field}"'
#    elif isinstance(field, bool):
#        return 'true' if field else 'false'
#    else:
#        try:
#            iterator = iter(field)
#            return '[' + ', '.join([field_to_string(i) for i in iterator]) + ']'
#        except TypeError:
#            pass
#        
#        return str(field)
#
#def config_to_string(config: dict) -> str:
#    s = "{\n"
#    lines = []
#    for key, val in config.items():
#        if key[:7] == 'zparams':
#            v = '[' + ', '.join([config_to_string(p).replace('\n', '').replace('\t', '').replace(',', ', ').replace('\'', '"') for p in val]) + ']'
#        else:
#            v = field_to_string(val)
#        lines.append(f"\t\"{key}\": {v}")
#    s += ',\n'.join(lines) + '\n}'

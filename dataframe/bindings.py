from abc import ABC, abstractmethod
from pathos.multiprocessing import ProcessingPool as Pool

import time
import tqdm
from functools import partial

from dataframe.dataframe_bindings import *


ATOL = 1e-6
RTOL = 1e-5


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

        self.num_threads = metadata.setdefault("num_threads", 1)
        self.num_threads_per_task = metadata.setdefault("num_threads_per_task", 1)
        self.atol = metadata.setdefault("atol", ATOL)
        self.rtol = metadata.setdefault("rtol", RTOL)
        self.average_congruent_runs = metadata.setdefault("average_congruent_runs", True)
        self.parallelization_type = metadata.setdefault("parallelization_type", self.SERIAL)
        self.record_error = metadata.setdefault("record_error", True)
        self.dataframe = DataFrame(self.atol, self.rtol)

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
            print(f'slide:\n {slide}')
            nruns = self.configs[i].get_nruns()
            for _ in range(1, nruns):
                idx += 1
                slide_tmp = results[idx]
                print(f'tpm:\n {slide_tmp}')
                if self.average_congruent_runs:
                    slide = slide.combine(slide_tmp, self.atol, self.rtol)
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
            print(f"Total runtime: {duration:0.0f}")

    @staticmethod
    def _do_run(config, num_threads):
        slide = config.compute(num_threads)
        slide.add_param(config.params)

        return slide

    def compute_serial(self, total_configs, verbose):
        if verbose:
            print(f"Computing in serial.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")

        results = []
        for i in tqdm.tqdm(range(len(total_configs))):
            results.append(ParallelCompute._do_run(total_configs[i], self.num_threads_per_task))

        return results

    def compute_pool(self, total_configs, verbose):
        if verbose:
            print(f"Computing in parallel. {self.num_threads} threads available.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")

        with Pool(self.num_threads) as pool:
            if verbose:
                results = list(tqdm.tqdm(pool.imap(partial(ParallelCompute._do_run, num_threads=self.num_threads_per_task), total_configs), total=len(total_configs)))
            else:
                results = list(pool.imap(partial(ParallelCompute._do_run, num_threads=self.num_threads_per_task), total_configs))

        return results


def load_data(filename: str) -> DataFrame:
    with open(filename, 'r') as f:
        s = f.read()

    return DataFrame(s)


def load_json(filename: str, verbose: bool = False) -> list:
    return parse_config(filename, verbose)


def write_config(params: list) -> str:
    return paramset_to_string(params)

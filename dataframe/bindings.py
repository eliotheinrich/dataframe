from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import concurrent

import os
import json
import time
import tqdm

from dataframe.dataframe_bindings import *


ATOL = 1e-6
RTOL = 1e-5


def save_config(config, filename):
    config = json.dumps(config, indent=1)
    config = config.replace('\\', '').replace(': false', ': 0').replace(': true', ': 1')
    with open(filename, 'w') as file:
        file.write(config)


def unbundle_params(param_bundle, p=None):
    params = []
    if p is None:
        param_bundle = param_bundle.copy()
        p = {}

    zparams = None
    for key in param_bundle:
        if key.startswith("zparams"):
            zparams = param_bundle[key]
            del param_bundle[key]
            break

    if zparams is not None:
        for zp in zparams:
            for key, val in zp.items():
                if key in param_bundle:
                    raise ValueError(f"Key {key} passed as a zipped parameter and an unzipped parameter; aborting.")

                p[key] = val
            params += unbundle_params(param_bundle.copy(), p.copy())

        return params

    scalar_keys = []
    vector_key = None
    for key, val in param_bundle.items():
        if hasattr(val, "__iter__") and not isinstance(val, str):
            vector_key = key
        else:
            p[key] = val
            scalar_keys.append(key)

    for key in scalar_keys:
        del param_bundle[key]

    if vector_key is None:
        params.append(p)
    else:
        vals = param_bundle[vector_key]
        del param_bundle[vector_key]
        for v in vals:
            p[vector_key] = v
            params += unbundle_params(param_bundle.copy(), p.copy())

    return params


class Config(ABC):
    def __init__(self, params):
        self.params = params.copy()
        self.num_runs = params.setdefault("num_runs", 1)

    def __getstate__(self):
        return self.params,

    def __setstate__(self, state):
        self.__init__(*state)

    def get_nruns(self):
        return int(self.params["num_runs"])

    @abstractmethod
    def compute(self, num_threads):
        pass

    @abstractmethod
    def clone(self):
        pass


class TimeConfig(Config):
    def __init__(self, params, simulator_generator, serialize=False):
        super().__init__(params)

        self.simulator_generator = simulator_generator
        self.simulator_driver = simulator_generator(params)
        self.serialize = serialize
        self._serialized_simulator = None

    # Allow injection of serialized simulator data
    def store_serialized_simulator(self, data):
        self._serialized_simulator = data

    def __getstate__(self):
        return self.params, self.simulator_generator, self.serialize, self._serialized_simulator

    def __setstate__(self, state):
        self.__init__(state[0], state[1], state[2])
        self.store_serialized_simulator(state[3])

    def compute(self, num_threads):
        self.simulator_driver.init_simulator(num_threads, self._serialized_simulator)

        slide = self.simulator_driver.generate_dataslide(self.serialize)

        self.params = self.simulator_driver.params
        return slide

    def clone(self):
        config = TimeConfig(self.params, self.simulator_generator, self.serialize)
        config.store_serialized_simulator(self._serialized_simulator)
        return config


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

        self.num_threads = int(metadata.setdefault("num_threads", 1))
        self.num_threads_per_task = int(metadata.setdefault("num_threads_per_task", 1))
        self.atol = float(metadata.setdefault("atol", ATOL))
        self.rtol = float(metadata.setdefault("rtol", RTOL))
        self.parallelization_type = int(metadata.setdefault("parallelization_type", self.SERIAL))
        self.average_congruent_runs = bool(metadata.setdefault("average_congruent_runs", True))
        self.batch_size = int(metadata.setdefault("batch_size", 1024))
        self.verbose = bool(metadata.setdefault("verbose", True))

        self.dataframe = DataFrame(self.atol, self.rtol)
        self.num_slides = None

    def compute(self):
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
            results = self.compute_serial(total_configs)
        elif self.parallelization_type == self.POOL:
            results = self.compute_pool(total_configs)

        if self.verbose:
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

        if self.verbose:
            print(f"Total runtime: {duration:0.0f}")

        return self.dataframe

    @staticmethod
    def _do_run(config, num_threads, id):
        try:
            _slide = config.compute(num_threads)

            if isinstance(_slide, DataSlide):
                slide = _slide
            else:
                slide = DataSlide(_slide)

            slide.add_param(config.params)

            return id, slide

        except Exception as e:
            if "SLURM_JOB_ID" in os.environ:
                filename = f"err_{os.environ['SLURM_JOB_ID']}_{id}.json"
            else:
                filename = f"err_{id}.json"

            print(f"Encountered an error; saving config params to {filename}!")
            save_config(config.params, filename)

            raise e

    def compute_serial(self, total_configs):
        if self.verbose:
            print("Computing in serial.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {len(total_configs)}")

        slides = [None for _ in range(self.num_slides)]
        if self.verbose:
            total_configs = tqdm.tqdm(total_configs)
        for i, config in total_configs:
            id, slide = ParallelCompute._do_run(config, self.num_threads_per_task, i)

            slides[id] = slide if slides[id] is None else slides[id].combine(slide, self.atol, self.rtol)

        return slides

    def compute_pool(self, total_configs):
        num_configs = len(total_configs)
        if self.verbose:
            print(f"Computing in parallel. {self.num_threads} threads available.")
            print(f"num_configs: {len(self.configs)}")
            print(f"total_runs: {num_configs}")

        slides = [None for _ in range(self.num_slides)]

        if self.verbose:
            progress = tqdm.tqdm(range(num_configs))
        else:
            progress = range(num_configs)
        with ProcessPoolExecutor(max_workers=self.num_threads) as pool:
            # Batch configs so that if num_configs is very large; ProcessPoolExecutor needs a copy of
            # total_configs for each process, so want to avoid creating num_configs**2 copies.
            num_batches = max(num_configs // self.batch_size, 1)
            last_batch_larger = num_configs % self.batch_size <= self.num_threads
            if not last_batch_larger:
                num_batches += 1

            for i in range(0, num_batches):
                i1 = i*self.batch_size
                if last_batch_larger and i == num_batches - 1:
                    i2 = num_configs
                else:
                    i2 = min((i+1)*self.batch_size, num_configs)

                futures = [pool.submit(ParallelCompute._do_run, config, self.num_threads_per_task, i) for i,config in total_configs[i1:i2]]
                completed_futures = concurrent.futures.as_completed(futures)

                for future in completed_futures:
                    id, slide = future.result()

                    slides[id] = slide if slides[id] is None else slides[id].combine(slide, self.atol, self.rtol)

                    if self.verbose:
                        progress.update(1)

        return slides

def compute(configs, **metadata):
    pc = ParallelCompute(configs, **metadata)
    return pc.compute()

def load_data(filename: str) -> DataFrame:
    extension = filename.split(".")[-1]
    if extension == "json":
        with open(filename, 'r') as f:
            s = f.read()
            return DataFrame(s)

    elif extension == "eve":
        with open(filename, 'rb') as f:
            s = bytes(f.read())
            return DataFrame(s)


def load_json(filename: str, verbose: bool = False) -> list:
    return parse_config(filename, verbose)


def write_param_bundle(params: list) -> str:
    return paramset_to_string(params)

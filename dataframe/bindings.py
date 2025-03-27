from concurrent.futures import ProcessPoolExecutor
import concurrent
import threading
import signal

import os
import json
import time
import tqdm

from abc import ABC, abstractmethod
from dataframe.dataframe_bindings import *


def terminating_thread(ppid):
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


ATOL = 1e-6
RTOL = 1e-5


class Config(ABC):
    def __init__(self, params):
        self.params = params
        self.num_threads = params.setdefault("num_threads", 1)

    def __getstate__(self):
        return self.params

    def __setstate__(self, args):
        self.__init__(*args)

    def get_buffer(self):
        raise RuntimeError("Called get_buffer on a Config which does not provide an implementation. Do not set serialize = True.")

    def inject_buffer(self, data):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def clone(self):
        pass


def register_component(component_generator, params, *args, **kwargs):
    if hasattr(component_generator, "create_and_emplace"):
        component, _params = component_generator.create_and_emplace(params, *args, **kwargs)
        for key, val in _params.items():
            if key not in params:
                params[key] = val
    else:
        component = component_generator(params, *args, **kwargs)

    return component


def save_param_matrix(param_matrix, filename):
    param_matrix = json.dumps(param_matrix, indent=1)
    param_matrix = param_matrix.replace('\\', '').replace(': false', ': 0').replace(': true', ': 1')
    with open(filename, "w") as file:
        file.write(param_matrix)


def load_param_matrix(filename):
    with open(filename, "r") as file:
        param_matrix = json.load(file)
    return param_matrix


class ZippedParams:
    def __init__(self, data):
        # TODO add some runtime checks for provided data
        self.data = data


def unbundle_param_matrix(param_bundle, p=None):
    params = []
    if p is None:
        param_bundle = param_bundle.copy()
        p = {}

    zipped_params = None
    for key, val in param_bundle.items():
        if isinstance(val, ZippedParams):
            zipped_params = param_bundle[key].data
            del param_bundle[key]
            break

    if zipped_params is not None:
        for zp in zipped_params:
            for key, val in zp.items():
                if key in param_bundle:
                    raise ValueError(f"Key {key} passed as a zipped parameter and an unzipped parameter; aborting.")

                p[key] = val
            params += unbundle_param_matrix(param_bundle.copy(), p.copy())

        return params

    scalar_keys = []
    vector_key = None
    for key, val in param_bundle.items():
        if hasattr(val, "__iter__") and not isinstance(val, str):
            vector_key = key
            param_bundle[key] = list(param_bundle[key])
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
            params += unbundle_param_matrix(param_bundle.copy(), p.copy())

    return params


class ParallelCompute:
    SERIAL = 0
    POOL = 1

    def __init__(self, configs, **metadata):
        self.configs = configs
        for config in self.configs:
            if not isinstance(config, Config):
                raise RuntimeError("compute accepts a list of Config.")
        self._metadata = metadata

        self.num_threads = int(metadata.setdefault("num_threads", 1))
        self.atol = float(metadata.setdefault("atol", ATOL))
        self.rtol = float(metadata.setdefault("rtol", RTOL))
        self.parallelization_type = int(metadata.setdefault("parallelization_type", self.POOL))
        self.average_congruent_runs = bool(metadata.setdefault("average_congruent_runs", True))
        self.batch_size = int(metadata.setdefault("batch_size", 1024))
        self.verbose = bool(metadata.setdefault("verbose", True))
        self.dump_errors = bool(metadata.setdefault("dump_errors", False))
        self.num_runs = int(metadata.setdefault("num_runs", 1))
        self.serialize = bool(metadata.setdefault("serialize", False))

        self.dataframe = DataFrame(self.atol, self.rtol)
        self.num_slides = None

    def average(self):
        return self.average_congruent_runs and not self.serialize

    def compute(self):
        start_time = time.time()

        total_configs = []
        j = 0
        for i, config in enumerate(self.configs):
            config.clone()

            for _ in range(self.num_runs):
                id = i if self.average() else j
                total_configs.append((id, config.clone()))
                j += 1

        self.num_slides = len(self.configs) if self.average() else len(total_configs)

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
        if self.average():
            self.dataframe.reduce()

        if self.verbose:
            print(f"Total runtime: {duration:0.0f}")

        return self.dataframe

    @staticmethod
    def _do_run(config, id, dump_errors, serialize):
        try:
            _slide = config.compute()

            if isinstance(_slide, DataSlide):
                slide = _slide
            else:
                slide = DataSlide(_slide)

            slide.add_param(config.params)

            if serialize:
                buffer = config.get_buffer()
                slide._inject_buffer(buffer)

            return id, slide

        except Exception as e:
            if dump_errors:
                if "SLURM_JOB_ID" in os.environ:
                    filename = f"err_{os.environ['SLURM_JOB_ID']}_{id}.json"
                else:
                    filename = f"err_{id}.json"

                print(f"Encountered an error; saving config params to {filename} and exiting!")
                save_config(config.params, filename)

            else:
                print(f"Encountered an error; exiting!")

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
            id, slide = ParallelCompute._do_run(config, i, self.dump_errors, self.serialize)

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

        with ProcessPoolExecutor(max_workers=self.num_threads, initializer=terminating_thread, initargs=(os.getpid(),)) as pool:
            # Batch configs; ProcessPoolExecutor needs a copy of every config (total_configs)
            # for each process, so want to avoid creating num_configs**2 copies.
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

                futures = [pool.submit(ParallelCompute._do_run, config, i, self.dump_errors, self.serialize) for i,config in total_configs[i1:i2]]
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

    else:
        with open(filename, 'rb') as f:
            s = bytes(f.read())
            frame = DataFrame(s)
            if "num_runs" in frame.params:
                num_runs = frame.params["num_runs"]
                frame.remove("num_runs")
                frame.metadata = {**frame.metadata, "num_runs": num_runs}
            return frame

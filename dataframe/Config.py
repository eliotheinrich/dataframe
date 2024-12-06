from abc import ABC, abstractmethod
from .bindings import DataSlide
import time

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


class CppConfig(Config):
    def __init__(self, params, _internal_config):
        super().__init__(params)
        self._internal_config = _internal_config

    def __getstate__(self):
        return self.params, self._internal_config

    def concretize(self):
        return self._internal_config(self.params)

    def compute(self, num_threads):
        config = self.concretize()
        return config.compute(num_threads)

    def clone(self):
        return CppConfig(self.params, self._internal_config)


class Simulator(ABC):
    @abstractmethod
    def __init__(self, params, num_threads):
        pass

    @abstractmethod
    def init(self, serialized_data):
        pass

    @abstractmethod
    def timesteps(self, num_steps):
        pass

    def equilibration_timesteps(self, num_steps):
        self.timesteps(num_steps)

    @abstractmethod
    def take_samples(self):
        pass

    @abstractmethod
    def serialize(self):
        pass


class SimulatorConfig(Config):
    def __init__(self, params, simulator_generator, serialize=False):
        super().__init__(params)

        self.simulator_generator = simulator_generator
        self.serialize = serialize
        self._serialized_simulator = None

        self.equilibration_timesteps = params.setdefault("equilibration_timesteps", 0)
        self.sampling_timesteps = params.setdefault("sampling_timesteps", 0)
        self.measurement_freq = params.setdefault("measurement_freq", 1)
        self.temporal_avg = params.setdefault("temporal_avg", False)

        self.save_samples = params.setdefault("save_samples", False)

        if self.temporal_avg and self.save_samples:
            raise RuntimeError("Cannot perform temporal average and save all samples.")

    # Allow injection of serialized simulator data
    def store_serialized_simulator(self, data):
        self._serialized_simulator = data

    def __getstate__(self):
        return self.params, self.simulator_generator, self.serialize, self._serialized_simulator

    def __setstate__(self, state):
        self.__init__(state[0], state[1], state[2])
        self.store_serialized_simulator(state[3])

    def compute(self, num_threads):
        start = time.time()
        slide = DataSlide()

        simulator = self.simulator_generator(self.params, num_threads)
        simulator.init(self._serialized_simulator)

        if self.sampling_timesteps == 0:
            num_timesteps = 0
            num_intervals = 1
        else:
            num_timesteps = self.measurement_freq
            num_intervals = self.sampling_timesteps // self.measurement_freq

        simulator.equilibration_timesteps(self.equilibration_timesteps)

        simulator.timesteps(num_timesteps)
        sample = simulator.take_samples()
        if self.save_samples:
            slide.add_samples(sample)
            slide.push_samples(sample)
        else:
            slide.add_data(sample)
            slide.push_samples_to_data(sample)

        for i in range(1, num_intervals):
            simulator.timesteps(num_timesteps)
            sample = simulator.take_samples()
            if self.save_samples:
                slide.push_samples(sample)
            else:
                slide.push_samples_to_data(sample, self.temporal_avg)

        end = time.time()
        duration = end - start
        slide.add_data("time")
        slide.push_samples_to_data("time", duration)

        if self.serialize:
            slide.buffer = simulator.serialize()

        return slide

    def clone(self):
        config = SimulatorConfig(self.params, self.simulator_generator, self.serialize)
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
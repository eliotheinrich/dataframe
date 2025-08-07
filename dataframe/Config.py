from abc import ABC, abstractmethod
import time

from .bindings import Config, DataSlide, register_component

class CppConfig(Config):
    def __init__(self, params, _internal_config):
        super().__init__(params)
        self._internal_config = _internal_config

    def __getstate__(self):
        return self.params, self._internal_config

    def concretize(self):
        return self._internal_config(self.params)

    def compute(self):
        config = self.concretize()
        return config.compute(self.num_threads)


class Simulator(ABC):
    @abstractmethod
    def __init__(self, params, num_threads):
        pass

    def init(self, serialized_data):
        pass

    def annealing_callback(self, num_steps, total_num_steps):
        pass

    @abstractmethod
    def timesteps(self, num_steps):
        pass

    def equilibration_timesteps(self, num_steps):
        self.timesteps(num_steps)

    @abstractmethod
    def take_samples(self):
        pass

    def serialize(self):
        raise NotImplementedError("serialize not implemented for this Simulator.")

    def get_texture(self):
        raise NotImplementedError("get_texture not implemented for this Simulator.")

    def key_callback(self, key):
        pass


class SimulatorConfig(Config):
    def __init__(self, params, simulator_generator):
        super().__init__(params)

        self.simulator_generator = simulator_generator
        self._serialized_simulator = None

        self.equilibration_timesteps = params.setdefault("equilibration_timesteps", 0)
        self.sampling_timesteps = params.setdefault("sampling_timesteps", 0)
        self.measurement_freq = params.setdefault("measurement_freq", 1)
        self.temporal_avg = params.setdefault("temporal_avg", False)
        self.epochs = params.setdefault("epochs", 1)
        self.annealing_timesteps = params.setdefault("annealing_timesteps", 0)

        self.simulator = None

    def get_buffer(self):
        return self.simulator.serialize()

    # Allow injection of serialized simulator data
    def inject_buffer(self, data):
        self._serialized_simulator = data

    def __getstate__(self):
        return self.params, self.simulator_generator, self._serialized_simulator

    def __setstate__(self, state):
        self.__init__(state[0], state[1])
        self.inject_buffer(state[2])

    def compute(self):
        start = time.time()
        slide = DataSlide()

        self.simulator = register_component(self.simulator_generator, self.params, self.num_threads)
        self.simulator.init(self._serialized_simulator) # If serialized data is available, use it

        if self.sampling_timesteps == 0:
            num_timesteps = 0
            num_intervals = 0
        else:
            num_timesteps = self.measurement_freq
            num_intervals = self.sampling_timesteps // self.measurement_freq

        def time_func(func, *args, **kwargs):
            t1 = time.time()
            return_val = func(*args, **kwargs)
            t2 = time.time()
            return return_val, t2 - t1

        sampling_time = 0.0
        steps_time = 0.0

        _, dt = time_func(self.simulator.equilibration_timesteps, self.equilibration_timesteps)
        steps_time += dt

        for j in range(self.epochs):
            self.simulator.annealing_callback(j, self.epochs)
            _, dt = time_func(self.simulator.timesteps, self.annealing_timesteps)
            steps_time += dt

            epoch_slide = DataSlide()

            for i in range(num_intervals):
                _, dt = time_func(self.simulator.timesteps, num_timesteps)
                steps_time += dt

                sample, dt = time_func(self.simulator.take_samples)
                sampling_time += dt

                if self.temporal_avg:
                    epoch_slide.add_data(sample)
                else:
                    slide.concat_data(sample)

            if self.temporal_avg:
                for key,val in epoch_slide.data.items():
                    slide.concat_data(key, val)



        end = time.time()
        duration = end - start
        slide.add_data("time", [duration])
        slide.add_data("sampling_time", [sampling_time])
        slide.add_data("steps_time", [steps_time])

        return slide

def get_timesteps(dataframe):
    import numpy as np
    keys = ["equilibration_timesteps", "sampling_timesteps", "measurement_freq", "annealing_timesteps", "epochs"]
    equilibration_timesteps, sampling_timesteps, measurement_freq, annealing_timesteps, epochs = dataframe.query(keys)

    num_points = sampling_timesteps // measurement_freq
    timesteps = np.array([])
    current_time = equilibration_timesteps

    for epoch in range(epochs):
        current_time += annealing_timesteps
        timesteps = np.concatenate((timesteps, np.arange(1, num_points + 1) * measurement_freq + current_time))
        current_time += sampling_timesteps 

    return np.array(timesteps)


class FuncConfig(Config):
    def __init__(self, params, function):
        super().__init__(params)
        self.function = function

    def compute(self):
        return self.function(self.params, self.num_threads)

    def __getstate__(self):
        return self.params, self.function

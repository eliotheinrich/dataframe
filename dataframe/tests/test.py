from dataframe import *
import numpy as np
import dill as pkl

import unittest

class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.frame = DataFrame()
        self.frame.add_metadata("tag", "exp42")
        self.frame.add_param("alpha", 1.2)

        slide = DataSlide()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        slide.add_param("p", 0.2)
        slide.add_param("promoted", 1)
        slide.add_data("test", data, [2, 3])
        slide.add_data("flat_test", np.random.rand(10));
        self.frame.add_slide(slide)

        slide = DataSlide()
        slide.add_param("p", 0.2)
        slide.add_param("promoted", 1)
        slide.add_data("test", data[::-1], [2, 3])
        slide.add_data("flat_test", np.random.rand(10));
        self.frame.add_slide(slide)

        slide = DataSlide()
        slide.add_param("p", "something else")
        slide.add_param("promoted", 1)
        slide.add_data("test", [0]*6, [2, 3])
        slide.add_data("flat_test", np.random.rand(10));
        self.frame.add_slide(slide)

        self.reduced_frame = DataFrame(self.frame)
        self.reduced_frame.reduce()

        self.slide = DataSlide()
        self.slide.add_param("alpha", 0.5)
        self.slide.add_param("beta", 1)
        self.slide.add_data("values", [10.0, 20.0, 30.0, 40.0], shape=[2, 2])

    def test_slide_basics(self):
        result1 = self.frame.query(["test"])
        self.assertEqual(result1.shape, (3, 2, 3))
    
        self.frame.reduce()
        result2 = self.frame.query(["test"])
        self.assertEqual(result2.shape, (2, 2, 3))
        self.assertEqual(np.allclose(result2[0], 3.5), True)

    def test_init_and_add_param(self):
        slide = DataSlide()
        slide.add_param("x", 42)
        self.assertIn("x", slide)
        self.assertEqual(slide["x"], 42)

    def test_access(self):
        self.assertEqual(len(self.frame.slides), 3)
        self.assertIn("alpha", self.frame)
        self.assertAlmostEqual(self.frame["alpha"], 1.2)
        self.assertEqual(self.frame["tag"], "exp42")

    def test_metadata(self):
        self.assertIn("tag", self.frame)
        self.assertEqual(self.frame["tag"], "exp42")

    def test_slide_pickle_roundtrip(self):
        pickled = pkl.dumps(self.slide)
        unpickled = pkl.loads(pickled)
        self.assertAlmostEqual(unpickled["alpha"], self.slide["alpha"])
        self.assertEqual(unpickled.get_data("values").shape, (2, 2))

    def test_frame_pickle_roundtrip(self):
        pickled = pkl.dumps(self.frame)
        unpickled = pkl.loads(pickled)

        self.assertEqual(len(unpickled.slides), len(self.frame.slides))
        self.assertEqual(unpickled.params, self.frame.params)
        self.assertEqual(unpickled.metadata, self.frame.metadata)

        t1, t2 = self.frame.query(["flat_test", "test"])
        t3, t4 = unpickled.query(["flat_test", "test"])
        self.assertTrue(np.allclose(t1, t3))
        self.assertTrue(np.allclose(t2, t4))

    def test_query_unique(self):
        p = self.frame.query(["p"], unique=True)
        self.assertEqual(len(p), 2)
        self.assertTrue("something else" in p and 0.2 in p)

    def test_query_mean(self):
        p, result1, result2 = self.frame.query(["p", "test", "flat_test"])
        self.assertEqual(result1.shape, (3, 2, 3))
        self.assertEqual(result2.shape, (3, 10))

        result1, result2 = self.reduced_frame.query(["test", "flat_test"])
        self.assertEqual(result1.shape, (2, 2, 3))
        self.assertEqual(result2.shape, (2, 10))

    def test_query_nsamples(self):
        result1, result2 = self.frame.query_nsamples(["test", "flat_test"])
        self.assertEqual(result1.shape, (3, 2, 3))
        self.assertEqual(result2.shape, (3, 10))
        self.assertTrue(np.allclose(result1, 1))
        self.assertTrue(np.allclose(result2, 1))

        result1, result2 = self.reduced_frame.query_nsamples(["test", "flat_test"])
        self.assertEqual(result1.shape, (2, 2, 3))
        self.assertEqual(result2.shape, (2, 10))
        self.assertTrue(np.allclose(result1[0], 2))
        self.assertTrue(np.allclose(result2[0], 2))
        self.assertTrue(np.allclose(result1[1], 1))
        self.assertTrue(np.allclose(result2[1], 1))

    def test_query_std(self):
        p, result1 = self.frame.query_std(["p", "test"])
        self.assertEqual(result1.shape, (3, 2, 3))
        self.assertTrue(np.allclose(result1, 0.0))

        result2 = self.reduced_frame.query_std(["test"])
        self.assertEqual(result2.shape, (2, 2, 3))


    def test_combine_and_reduce(self):
        other = DataFrame(self.frame)

        slide = DataSlide()
        slide.add_param("p", "something else")
        slide.add_param("promoted", 1)
        slide.add_data("test", [1]*6, [2, 3])
        slide.add_data("flat_test", np.random.rand(10))
        other.add_slide(slide)

        new_frame = other + self.frame
        with self.assertRaises(KeyError):
            p = new_frame["promoted"]

        new_frame.reduce()

        p = new_frame["promoted"]
        self.assertEqual(p, 1)

        p, r1, r2 = new_frame.query(["p", "test", "flat_test"])

        self.assertEqual(r1.shape, (2, 2, 3))
        self.assertEqual(r2.shape, (2, 10))

        self.assertTrue(np.allclose(r1[0], 3.5))
        self.assertTrue(np.allclose(r1[1], 1/3))

    def test_promote_params(self):
        with self.assertRaises(KeyError):
            p = self.frame["promoted"]
        self.frame.promote_params()

        p = self.frame["promoted"]
        self.assertEqual(p, 1)


class SimpleSimulator(Simulator):
    def __init__(self, params, num_threads):
        super().__init__(params, num_threads)
        self.p = params["p"]
        self.t = params["t"]

        self.steps_done = 0

        np.random.seed(params["seed"])

    def timesteps(self, num_timesteps):
        self.steps_done += num_timesteps

    def take_samples(self):
        samples = {"steps_finished": [self.p*self.steps_done, self.t*self.steps_done, np.random.rand()]}
        return samples


class TestSimulator(unittest.TestCase):
    def setUp(self):
        params = {"p": 0.5, "t": 2.0, "seed": 314, "sampling_timesteps": 100, "equilibration_timesteps": 100, "measurement_freq": 2, "temporal_avg": False}

        self.config = SimulatorConfig(params, SimpleSimulator)
        self.slide = self.config.compute()
        self.frame1 = DataFrame()
        self.frame1.add_param(params)
        self.frame1.add_slide(self.slide)

        params = {"p": 0.5, "t": 2.0, "seed": 314, "sampling_timesteps": 100, "equilibration_timesteps": 100, "measurement_freq": 2, "temporal_avg": True}

        self.config = SimulatorConfig(params, SimpleSimulator)
        self.slide = self.config.compute()
        self.frame2 = DataFrame()
        self.frame2.add_param(params)
        self.frame2.add_slide(self.slide)

    def test_simulator_results_temporal(self):
        s = self.frame1.query(["steps_finished"])

        t = get_timesteps(self.frame1)

        expected_s1 = self.frame1["p"] * t
        expected_s2 = self.frame1["t"] * t

        self.assertTrue(np.allclose(expected_s1, s[0,:,0]))
        self.assertTrue(np.allclose(expected_s2, s[0,:,1]))

        # TODO multiple runs?

    def test_simulator_results_avg(self):
        s1 = self.frame1.query(["steps_finished"])
        s2 = self.frame2.query(["steps_finished"])
        s2_std = self.frame2.query_std(["steps_finished"])
        s2_nsamples = self.frame2.query_nsamples(["steps_finished"])

        for i in range(s1.shape[2]):
            self.assertAlmostEqual(np.mean(s1[0,:,i]), s2[0,i])
            self.assertAlmostEqual(np.std(s1[0,:,i], ddof=1), s2_std[0,i])
            self.assertEqual(len(s1[0,:,i]), s2_nsamples[0,i])


class TestConfig(Config):
    def __init__(self, params):
        super().__init__(params)
        self.p = params["p"]
        self.num_samples = params["n"]
        #self.sampler = register_component(TestSampler, params)

    @profiler
    def compute(self):
        r = self.p*np.random.rand(self.num_samples)

        slide = DataSlide()
        for i in range(self.num_samples):
            slide.add_data("r", [r[i]])

        return slide


class TestParallelCompute(unittest.TestCase):
    def setUp(self):
        self.params_matrix = {"p": np.arange(0.0, 100.0, 0.5), "n": 1000}
        self.configs = [TestConfig(p) for p in unbundle_param_matrix(self.params_matrix)]

    @profiler
    def test_compute_serial(self):
        nruns = 10
        frame = compute(self.configs, num_threads=1, parallelization_type=0, num_runs=nruns, verbose=False)
        keys = ["p", "r"]

        p, r = frame.query(keys)
        p = np.array(p)
        self.assertTrue(np.all((np.abs(r[:,0] - p/2)) < 1.0))

        _, r_nsamples = frame.query_nsamples(keys)
        self.assertTrue(np.allclose(r_nsamples, nruns * self.params_matrix["n"]))

    def test_compute_parallel(self):
        nruns = 10
        frame = compute(self.configs, num_threads=4, parallelization_type=1, num_runs=nruns, verbose=False)
        keys = ["p", "r"]

        p, r = frame.query(keys)
        p = np.array(p)
        self.assertTrue(np.all((np.abs(r[:,0] - p/2)) < 1.0))

        _, r_nsamples = frame.query_nsamples(keys)
        self.assertTrue(np.allclose(r_nsamples, nruns * self.params_matrix["n"]))

    @classmethod
    def tearDownClass(cls):  
        profiler.print_stats() 
    
if __name__ == "__main__":
    unittest.main()

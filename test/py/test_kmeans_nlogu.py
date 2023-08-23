import unittest
import _kmeans_nlogu
import _ckmeans_1d_dp
import numpy as np
import bisect


class TestKMeansNLogU(unittest.TestCase):

    def test_rmse_is_adequate(self):
        n_elements = 1000000
        x = np.random.default_rng().normal(loc=0, scale=1000, size=(n_elements, 1))
        max_rmses = np.array([3.5])
        borders = np.empty(shape=(n_elements + 1,))
        centers = np.empty(shape=(n_elements,))
        kOpts = np.empty(shape=(1,), dtype=np.int64)
        res_rmses = np.empty_like(max_rmses)
        _kmeans_nlogu.kmeans_nlogu(x, max_rmses, borders, kOpts, res_rmses, centers)
        borders = np.flip(borders[:kOpts[0]], axis=0)
        centers = np.flip(centers[:kOpts[0]], axis=0)

        restored = np.array(list(map(lambda e: [centers[bisect.bisect_left(borders[:n_elements], e)]], x)))
        rmse = np.sqrt(np.mean((restored - x) ** 2, axis=0))
        with self.subTest(expected=rmse[0], actual=res_rmses[0]):
            self.assertTrue(np.isclose(rmse[0], res_rmses[0], rtol=1e-2, atol=1e-5))  # replace with np.isclose

    def test_close_enough_rmse_for_random(self):
        n_elements = 1000000
        x = np.random.default_rng().uniform(low=-10000, high=10000, size=(n_elements, 1))
        expected_rmse = 10.0
        max_rmses = np.array([expected_rmse])
        borders = np.empty(shape=(n_elements + 1,))
        centers = np.empty(shape=(n_elements,))
        kOpts = np.empty(shape=(1,), dtype=np.int64)
        res_rmses = np.empty_like(max_rmses)
        _kmeans_nlogu.kmeans_nlogu(x, max_rmses, borders, kOpts, res_rmses, centers)

        with self.subTest(expected=expected_rmse, actual=res_rmses[0]):
                self.assertTrue(np.isclose(expected_rmse, res_rmses[0], rtol=0.1, atol=1e-3))

    def test_identical_results_with_linear(self):
        n_elements = 100
        x = np.random.default_rng().normal(loc=0, scale=1, size=(n_elements, 1))
        x_linear = x.copy().T
        expected_rmse = 0.4
        max_rmses = np.array([expected_rmse])
        borders = np.empty(shape=(n_elements + 1,))
        centers_nlogu = np.empty(shape=(n_elements,))
        centers_linear = np.zeros_like(centers_nlogu)
        kOpts = np.empty(shape=(1,), dtype=np.int64)
        kOpts_linear = np.empty_like(kOpts)
        res_rmses = np.empty_like(max_rmses)
        _kmeans_nlogu.kmeans_nlogu(x, max_rmses, borders, kOpts, res_rmses, centers_nlogu)
        _ckmeans_1d_dp.ckmeans(x_linear,0, centers_linear, kOpts_linear, 8, kOpts[0])
        centers_nlogu = np.flip(centers_nlogu[:kOpts[0]], axis=0)
        centers_linear = centers_linear[:kOpts[0]]
        with self.subTest(linear=centers_linear, nlogu=centers_nlogu):
            self.assertTrue(np.isclose(centers_linear, centers_nlogu).all())


if __name__ == '__main__':
    unittest.main()

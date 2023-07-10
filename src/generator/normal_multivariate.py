from numpy.random import default_rng
from generator import Generator


class NormalMultivariateDistributionGenerator(Generator):
    name = "Normal multivariate distribution RNG"

    def gen(self, mean=(0,), cov=((1,),), output_shape=None):
        return default_rng().multivariate_normal(mean, cov, output_shape)

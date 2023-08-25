from numpy.random import default_rng
from stil_internship.generator import Generator


class NormalDistributionGenerator(Generator):
    name = "Normal distribution RNG"

    def gen(self, mean=0, dev=1, output_shape=None):
        return default_rng().normal(mean, dev, output_shape)

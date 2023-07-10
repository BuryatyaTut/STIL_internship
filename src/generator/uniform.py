from numpy.random import default_rng
from generator import Generator


class UniformGenerator(Generator):
    name = "Uniform distribution RNG"

    def gen(self, low=0, high=1, output_shape=None):
        return default_rng().uniform(low, high, output_shape)

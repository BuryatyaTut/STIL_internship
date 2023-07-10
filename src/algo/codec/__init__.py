import abc as __abc


class CompressionAlgorithm(__abc.ABC):
    def __init__(self):
        ...

    @property
    @__abc.abstractmethod
    def name(self):
        ...

    @__abc.abstractmethod
    def compress(self, table_file_path, compressed_file_path):
        ...

    @__abc.abstractmethod
    def decompress(self, compressed_file_path, decompressed_file_path):
        ...


class LearningCompressionAlgorithm(CompressionAlgorithm):

    @__abc.abstractmethod
    def fit(self, training_data):
        ...


class NonLearningCompressionAlgorithm(CompressionAlgorithm):
    ...


class LosslessCompressionAlgorithm(CompressionAlgorithm):
    ...


class LossyCompressionAlgorithm(CompressionAlgorithm):
    ...
import abc


class CompressionAlgorithm(abc.ABC):
    def __init__(self):
        ...

    @property
    @abc.abstractmethod
    def name(self):
        ...

    @abc.abstractmethod
    def compress(self, table_file_path, compressed_file_path):
        ...

    @abc.abstractmethod
    def decompress(self, compressed_file_path, decompressed_file_path):
        ...


class LearningCompressionAlgorithm(CompressionAlgorithm):

    @abc.abstractmethod
    def fit(self, training_data):
        ...


class NonLearningCompressionAlgorithm(CompressionAlgorithm):
    ...


class LosslessCompressionAlgorithm(CompressionAlgorithm):
    ...


class LossyCompressionAlgorithm(CompressionAlgorithm):
    ...
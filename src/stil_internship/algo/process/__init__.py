import abc


class Processing(abc.ABC):
    def __init__(self):
        ...

    @property
    @abc.abstractmethod
    def name(self):
        ...

    @abc.abstractmethod
    def do_preprocess(self, raw_file_path, processed_file_path):
        ...

    @abc.abstractmethod
    def do_postprocess(self, decompressed_file_path, postprocessed_file_path):
        ...

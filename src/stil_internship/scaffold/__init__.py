import inspect
import os
import pickle
import random
import time
from stil_internship.algo.codec import LosslessCompressionAlgorithm, LearningCompressionAlgorithm
import pandas as pd
import numpy as np

def get_compression_rate(input_file, compressed_file):  # I'm assuming compressed_table is actually bytes-like
    input_file.seek(0, os.SEEK_END)
    compressed_file.seek(0, os.SEEK_END)
    print(compressed_file.name, input_file.tell(), compressed_file.tell())
    return compressed_file.tell() / input_file.tell()


def get_loss_rate(input_file_path, decompressed_file_path):
    with open(input_file_path, 'rb') as input_file, open(decompressed_file_path, 'rb') as decompressed_file:
        src = pickle.load(input_file).select_dtypes(["number"]).to_numpy()
        dec = pickle.load(decompressed_file).select_dtypes(["number"]).to_numpy()
        return np.sqrt(np.mean((src - dec) ** 2))




class Scaffold:
    def __init__(self, algorithm, processing, table_path):
        self.algorithm = algorithm
        self.processing = processing
        self.table_path = table_path
        self.logs = False

        self.benchmark = {"name": type(self.algorithm).__name__, "table_path": table_path}

    def start(self, run_type="test", logs=False):
        self.logs = logs

        if self.logs:
            if isinstance(self.algorithm, LosslessCompressionAlgorithm):
                self.benchmark["isLossLess"] = True
            else:
                self.benchmark["isLossLess"] = False

            if isinstance(self.algorithm, LearningCompressionAlgorithm):
                self.benchmark["isLearning"] = True
            else:
                self.benchmark["isLearning"] = False

        if run_type == 'test':
            self.test()
        elif run_type == 'compress':
            self.preprocess_and_compress()
        elif run_type == 'decompress':
            self.decompress_and_postprocess()

    def measure_time(flag):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()

                if getattr(self, flag):
                    print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
                    self.benchmark[func.__name__ + "_time"] = end_time - start_time

                return result

            return wrapper

        return decorator

    def get_file_names(self):
        return {"compressed": self.table_path + "_compressed",
                "decompressed": self.table_path + "_decompressed",
                "preprocessed": self.table_path + "_preprocessed",
                "postprocessed": self.table_path + "_postprocessed"}

    @measure_time("logs")
    def preprocess_and_compress(self):
        filenames = self.get_file_names()

        self.processing.do_preprocess(self.table_path, filenames['preprocessed'])
        self.algorithm.compress(filenames["preprocessed"], filenames["compressed"])

        if self.logs:
            compression_rate = get_compression_rate_2(filenames["preprocessed"], filenames["compressed"])

            self.benchmark["compression_rate"] = compression_rate
        return 0

    @measure_time("logs")
    def decompress_and_postprocess(self):
        filenames = self.get_file_names()

        self.algorithm.decompress(filenames["compressed"], filenames["decompressed"])

        self.processing.do_postprocess(filenames["decompressed"], filenames["postprocessed"])

        if self.logs:
            loss_rate = get_loss_rate(filenames["preprocessed"], filenames["decompressed"])
            self.benchmark["loss_rate"] = loss_rate

        return 0

    @measure_time("logs")
    def test(self):
        self.preprocess_and_compress()
        self.decompress_and_postprocess()
        return 0  # TODO: result codes


def get_compression_rate_2(original_file_path, compressed_file_path):
    with open(original_file_path, 'rb') as input_file, open(compressed_file_path, 'rb') as compressed_file:
        input_file.seek(0, os.SEEK_END)
        compressed_file.seek(0, os.SEEK_END)
        print(compressed_file.name, input_file.tell(), compressed_file.tell())
        return compressed_file.tell() / input_file.tell()

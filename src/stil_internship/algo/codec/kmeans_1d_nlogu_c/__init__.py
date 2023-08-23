import pickle
import warnings

import numpy as np
import _kmeans_nlogu

import pandas as pd
import zstandard as zstd
from algo.codec import NonLearningCompressionAlgorithm, LossyCompressionAlgorithm


class KMeansLogLinearCDTO:
    def __init__(self, non_numeric, order, order_numeric, index, quantized_columns):
        self.non_numeric = non_numeric
        self.order = order
        self.order_numeric = order_numeric
        self.index = index
        self.quantized_columns = quantized_columns


class KMeansLogLinearCCompression(NonLearningCompressionAlgorithm, LossyCompressionAlgorithm):
    name = "O(n log U) KMeans"

    def __init__(self, max_rmses, raw_compression_ratio=1, threads=1):
        super().__init__()
        warnings.warn("This class is broken for the sake of being fit to exact task.")
        self.raw_compression_ratio = raw_compression_ratio
        self.max_rmses = max_rmses
        self.threads = threads

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            df = pickle.load(table_file)
            numeric = df.select_dtypes(['number']).astype(np.double)
            non_numeric = df.select_dtypes(exclude=['number'])
            order = df.columns
            order_numeric = numeric.columns
            index = df.index
            numeric_np = numeric.to_numpy().T
            # print(numeric)
            # print(numeric.shape)

            centers = np.zeros(shape=(numeric_np.shape[0], numeric_np.shape[1]), dtype=np.double)
            kopts = np.zeros(shape=(numeric_np.shape[0],), dtype=np.int64)
            _kmeans_nlogu.kmeans_nlogu(numeric_np, self.max_rmses,
                                       centers, kopts, self.threads)

            quantized_columns = [{"centers": centers[i, :kopts[i]]} for i in
                                 range(numeric_np.shape[0])]
            print('quantized columns:')
            print(quantized_columns)
            output = KMeansLinearCDTO(non_numeric, order, order_numeric, index, quantized_columns)
            file = pickle.dumps(output)

            compressor = zstd.ZstdCompressor(level=22)
            compressed_data = compressor.compress(file)
            compressed_file.write(compressed_data)
            compressed_file.flush()

    def decompress(self, compressed_file_path, decompressed_file_path):
        with open(compressed_file_path, 'rb') as compressed_file, open(decompressed_file_path,
                                                                       'wb') as decompressed_file:
            decompressor = zstd.ZstdDecompressor()
            decompressed = decompressor.decompress(compressed_file.read())
            from_file = pickle.loads(decompressed)
            quantized_columns = from_file.quantized_columns
            restored_columns = []
            for column in quantized_columns:
                print(column['cluster'])
                restored_columns.append(np.array(column['centers'])[column['cluster']])
            restored_columns = np.stack(restored_columns, axis=1)
            restored_numeric_df = pd.DataFrame(data=restored_columns, index=from_file.index,
                                               columns=from_file.order_numeric)
            restored = pd.merge(from_file.non_numeric, restored_numeric_df, left_index=True, right_index=True)[
                from_file.order]
            print(restored.head())
            pickle.dump(restored, decompressed_file)

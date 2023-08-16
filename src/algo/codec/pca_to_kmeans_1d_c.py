import pickle

import _ckmeans_1d_dp
import zstandard as zstd
import pandas as pd
import sklearn
import sklearn.decomposition
from algo.codec import NonLearningCompressionAlgorithm, LossyCompressionAlgorithm
import numpy as np

class PCAToKMeansCompression(NonLearningCompressionAlgorithm, LossyCompressionAlgorithm):
    name = "PCA -> KMeans"
    '''
        NOTE:
        There is no reasonable way to control any metric yet. It might work if we tell KMeans what the components are.
    '''
    def __init__(self, n_pca_components='mle', kmeans_component_ratio=1):
        super().__init__()
        self.n_components = n_pca_components
        self.raw_compression_ratio = kmeans_component_ratio

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            df = pickle.load(table_file)
            numeric = df.select_dtypes(['number'])
            non_numeric = df.select_dtypes(exclude=['number'])
            order = df.columns
            order_numeric = numeric.columns
            index = df.index
            pca = sklearn.decomposition.PCA(n_components=self.n_components)
            transformed = pca.fit_transform(numeric).T
            parameters = pca.get_params()
            components = pca.components_
            # restored = pca.inverse_transform(transformed)
            mean = pca.mean_

            quantized_columns = []
            for column in transformed:
                clusters = np.zeros(shape=(column.size,), dtype=np.int64)
                centers = np.zeros(shape=(column.size,), dtype=np.longdouble)
                column = np.asarray(column.astype(np.longdouble))
                kOpt = _ckmeans_1d_dp.ckmeans(column, 0,
                                              np.ceil(column.size / self.raw_compression_ratio).astype(int),
                                              clusters, centers)
                # kmeans = KMeansLinear(column, np.ceil(column.size / self.raw_compression_ratio), self.max_rmse)

                # result = kmeans.run()
                result = {"cluster": clusters, "centers": centers[:kOpt]}
                quantized_columns.append(result)
                del result
            print("transformed head:", transformed[0])
            print("pca loadings: ", pca.components_.T)
            print("pca scores: ", pca.score(numeric), pca.score_samples(numeric))
            # print('restored: ', restored[0])
            print('src ', numeric.iloc[0])
            to_file = (quantized_columns, non_numeric, order, parameters, components, mean, order_numeric, index)
            file = pickle.dumps(to_file)
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
            quantized_columns = from_file[0]
            restored_columns = []
            for column in quantized_columns:
                restored_columns.append(np.array(column['centers'])[column['cluster']])
            restored_columns = np.stack(restored_columns, axis=1)
            print(restored_columns)
            non_numeric = from_file[1]
            order = from_file[2]
            parameters = from_file[3]
            components = from_file[4]
            mean = from_file[5]
            order_numeric = from_file[6]
            index = from_file[7]
            pca = sklearn.decomposition.PCA()
            pca.set_params(**parameters)
            pca.components_ = components
            pca.mean_ = mean
            restored_numeric = pca.inverse_transform(restored_columns)
            restored_numeric_df = pd.DataFrame(data=restored_numeric, index=index, columns=order_numeric)
            restored = pd.merge(non_numeric, restored_numeric_df, left_index=True, right_index=True)[order]
            pickle.dump(restored, decompressed_file)

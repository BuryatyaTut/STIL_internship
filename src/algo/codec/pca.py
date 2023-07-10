import pickle
import zstandard as zstd
import sklearn
import pandas as pd
from algo.codec import NonLearningCompressionAlgorithm, LossyCompressionAlgorithm


class PCACompression(NonLearningCompressionAlgorithm, LossyCompressionAlgorithm):
    name = "sklearn PCA + MLE from pickle"

    def __init__(self, n_components='mle'):
        super().__init__()
        self.n_components = n_components

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            df = pickle.load(table_file)
            numeric = df.select_dtypes(['number'])
            non_numeric = df.select_dtypes(exclude=['number'])
            order = df.columns
            order_numeric = numeric.columns
            index = df.index
            pca = sklearn.decomposition.PCA(n_components=self.n_components, svd_solver='full')
            transformed = pca.fit_transform(numeric)
            print("transformed head:", transformed[0])
            print("pca scores: ", pca.score(numeric), pca.score_samples(numeric))
            parameters = pca.get_params()
            components = pca.components_
            restored = pca.inverse_transform(transformed)
            mean = pca.mean_
            print('restored: ', restored[0])
            print('src ', numeric.iloc[0])
            to_file = (transformed, non_numeric, order, parameters, components, mean, order_numeric, index)
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
            transformed = from_file[0]
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
            restored_numeric = pca.inverse_transform(transformed)
            restored_numeric_df = pd.DataFrame(data=restored_numeric, index=index, columns=order_numeric)
            restored = pd.merge(non_numeric, restored_numeric_df, left_index=True, right_index=True)[order]
            pickle.dump(restored, decompressed_file)

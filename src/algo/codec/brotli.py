import brotli
from algo.codec import LosslessCompressionAlgorithm, NonLearningCompressionAlgorithm


class BrotliLibraryCompressionAlgorithm(LosslessCompressionAlgorithm, NonLearningCompressionAlgorithm):
    name = 'brotli library'

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            compressed_file.write(brotli.compress(table_file.read()))
        return 0

    def decompress(self, compressed_file_path, decompressed_file_path):
        with open(compressed_file_path, 'rb') as compressed_file, \
                open(decompressed_file_path, 'wb') as decompressed_file:
            decompressed_file.write(brotli.decompress(compressed_file.read()))
        return 0

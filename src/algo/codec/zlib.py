import zlib

from algo.codec import NonLearningCompressionAlgorithm, LosslessCompressionAlgorithm


class ZlibCompression(NonLearningCompressionAlgorithm, LosslessCompressionAlgorithm):
    name = 'zlib'

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as f_in:
            with open(compressed_file_path, 'wb') as f_out:
                data = f_in.read()
                compressed_data = zlib.compress(data)
                f_out.write(compressed_data)

    def decompress(self, compressed_file_path, decompressed_file_path):
        with open(compressed_file_path, 'rb') as f_in:
            with open(decompressed_file_path, 'wb') as f_out:
                data = f_in.read()
                decompressed_data = zlib.decompress(data)
                f_out.write(decompressed_data)

from algo.codec import LosslessCompressionAlgorithm, NonLearningCompressionAlgorithm
import zstandard as zstd


class ZstdLibraryCompressionAlgorithm(LosslessCompressionAlgorithm, NonLearningCompressionAlgorithm):
    name = 'zstd library'

    def compress(self, table_file_path, compressed_file_path):
        with open(table_file_path, 'rb') as table_file, open(compressed_file_path, 'wb') as compressed_file:
            compressor = zstd.ZstdCompressor(level=22, threads=-1)
            compressor.copy_stream(table_file, compressed_file)
        return 0

    def decompress(self, compressed_file_path, decompressed_file_path):
        with open(compressed_file_path, 'rb') as compressed_file, \
                open(decompressed_file_path, 'wb') as decompressed_file:
            decompressor = zstd.ZstdDecompressor()
            decompressor.copy_stream(compressed_file, decompressed_file)
        return 0

import shutil

from algo.codec import LosslessCompressionAlgorithm, NonLearningCompressionAlgorithm


class DummyCompressionAlgorithm(LosslessCompressionAlgorithm, NonLearningCompressionAlgorithm):
    name = 'Dummy Compression Algorithm'

    def compress(self, table_file_path, compressed_file_path):
        if table_file_path == compressed_file_path:
            return 0
        shutil.copy2(table_file_path, compressed_file_path)
        print("I am dummy compression")
        return 0  # TODO: status codes

    def decompress(self, compressed_file_path, decompressed_file_path):
        if compressed_file_path == decompressed_file_path:
            return 0
        shutil.copy2(compressed_file_path, decompressed_file_path)
        print("I am dummy decompression")
        return 0  # TODO: status codes

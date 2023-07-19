import pickle
import pandas as pd
from algo.process import Processing


class NormalizeProcessing(Processing):
    name = 'binary'

    def do_preprocess(self, raw_file_path, processed_file_path):

        df = pd.read_csv(raw_file_path, index_col=0)
        self.mean = df.mean()
        self.std = df.std()
        df = (df - self.mean) / self.std
        with open(processed_file_path, 'wb') as processed_file:
            pickle.dump(df, processed_file)

    def do_postprocess(self, decompressed_file_path, postprocessed_file_path):
        with open(decompressed_file_path, 'a+b') as decompressed_file, \
                open(postprocessed_file_path, 'wb') as postprocessed_file:
            decompressed_file.seek(0)
            df = pickle.load(decompressed_file)
            df = df * self.std + self.mean
            df.to_csv(postprocessed_file)
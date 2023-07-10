import shutil

from algo.process import Processing


class DummyProcessing(Processing):
    name = "Dummy Pre/Post-Processing"

    def do_preprocess(self, raw_file_path, processed_file_path):
        if raw_file_path == processed_file_path:
            return 0
        shutil.copy2(raw_file_path, processed_file_path)
        return 0  # TODO: replace with enum / status codes

    def do_postprocess(self, processed_file_path, postprocessed_file_path):
        return 0  # TODO: replace with enum/status codes

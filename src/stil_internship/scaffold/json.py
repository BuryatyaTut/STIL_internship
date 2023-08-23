import json
from scaffold import Scaffold


def json_scaffold(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    scaffold = Scaffold(globals()[config["Algorithm"]](), globals()[config["Processing"]](), config["table_path"])
    scaffold.start(config["run_type"], config["logs"])


# loads the config yaml

import os
import yaml

def load_config(file_path = "configs/pretrain_config.yaml"):
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(script_dir, file_path)

    with open(abs_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
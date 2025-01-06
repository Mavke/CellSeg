import yaml
import os
from train_scripts.train_multigpu import main

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--config_files', type=str, help='Path to the config files', default='./pannuke_config')

    args = parser.parse_args()

    for root, _, filenames in os.walk(args.config_files):
            for filename in filenames:
                  with open(os.path.join(root, filename)) as file:
                        config = yaml.safe_load(file)
                        main(config)
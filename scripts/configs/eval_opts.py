import argparse
import os
import yaml
from meri_evaluation.utils import load_yaml



def parse_args():

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--config_file_path', type=str, required=True)

    # Parse the argument
    args = parser.parse_args()

    return args

def load_eval_configuration(config_file_path: str):

    assert os.path.exists(config_file_path)

    config = load_yaml(config_file_path)

    gt_int_format = False
    if "gt_int_format" in config.keys():
        gt_int_format = config["gt_int_format"]

    return config["dataset_path"], config["res_dir"], config["cache_dir"], config["methods_to_run"], gt_int_format

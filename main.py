import os
import argparse
import yaml
import h5py
import numpy as np
import torch
import gymnasium as gym 
from datetime import datetime
from model.random import BasicModel
from collector.data_collector import DataCollector, DataCollector_MiniGrid
import wandb

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print("Loaded configuration:", config)
    return config

def initialize_wandb(config):
    wandb.init(project=config['wandb']['project_name'],
               entity=config['wandb']['entity'],
               config=config)

def get_timestamp():
    """Returns a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(path):
    """Ensure directory exists, if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/random_minigrid.yaml', help='Path to the configuration file (default: config/random.yaml)')
    args = parser.parse_args()

    config = load_config(args.config)
    if config['wandb']['load'] == True:
        initialize_wandb(config)   

    env_name = config['environment']['name']
    max_path_length = config['environment']['max_path_length']
    num_data_points = config['environment']['num_data_points']
    model_name = config['model']['name']
    random_policy = config['model']['random_policy']
    model = BasicModel(env_name)
    collector = DataCollector_MiniGrid(config, model, max_path_length, num_data_points, random=random_policy)
    data = collector.collect_data()

    # Generate a unique file name using the timestamp
    timestamp = get_timestamp()
    file_prefix = "data"
    directory_path = f"data/{env_name}/{model_name}"
    ensure_directory(directory_path)
    output_file = f"{directory_path}/{env_name}_{model_name}_{num_data_points}_{timestamp}.hdf5"
    compression = config['output']['compression']

    hfile = h5py.File(output_file, 'w')
    for k, v in data.items():
        hfile.create_dataset(k, data=v, compression=compression)
    hfile.close()

    print(f"Data collected and saved successfully in {output_file}.")

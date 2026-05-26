import argparse
import yaml
import torch
import numpy as np
import random
import os

from trainer.train import train_model
from data.data_controller import split_filename

def main():
    """
    Main function to run the training process.

    This function parses command-line arguments for the configuration file and
    seed, sets up the environment, loads the configuration, and starts the
    training process by calling `train_model`.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        type=str,
        default="debug",
        help="Configuration file name in ./configs/ directory."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    config_dir = f"configs/{args.config}.yaml"
    
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)

    config['output_path'] = f"./results/{config['model_name'].replace('/', '__')}/{config['run_name']}"
    if os.environ.get("LOCAL_RANK", "0") == "0":
        assert not os.path.isdir(config['output_path']), "Model directory already exists. Please change the run name."
    
        os.makedirs(config['output_path'], exist_ok=True)
        with open(f"{config['output_path']}/config.yaml", "w") as f:
            yaml.safe_dump(config, f)
    
    config['train_file'] = [file_path for file_path in split_filename(config['train_file'])]
    for file in config['train_file']:
        assert os.path.isfile(file), f"Train file not found.\n Please check the path: {file}"
        
    train_model(config)
        
if __name__ == "__main__":
    main()
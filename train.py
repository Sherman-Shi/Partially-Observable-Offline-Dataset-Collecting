import yaml
import gym
import wandb
from pathlib import Path
import torch 
import sys 
sys.path.append("/home/zhenpeng/桌面/PORL/Partially-Observable-Offline-Dataset-Collecting")

import torchkit.pytorch_utils as ptu
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from trainer.trainer import Trainer

def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def main():
    # Load configuration
    config_path = Path("/home/zhenpeng/桌面/PORL/Partially-Observable-Offline-Dataset-Collecting/config/train/config.yaml")
    config = load_config(config_path)

    # Initialize environment
    env = gym.make(config['env']['name'])
    ptu.set_gpu_mode(torch.cuda.is_available(), 0)

    # Dynamically get observation and action dimensions
    obs_dim = env.observation_space.shape[0]
    # Handle both discrete and continuous action spaces
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n  # Use n for discrete action spaces
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]  # Use shape for continuous action spaces
    else:
        raise NotImplementedError("Action space type not supported")
        
    # Initialize model
    agent = Policy_RNN(
        obs_dim=obs_dim,
        action_dim=action_dim,
        encoder=config['model']['encoder'],
        algo_name=config['model']['algo'],
        action_embedding_size=config['model']['action_embedding_size'],
        observ_embedding_size=config['model']['observ_embedding_size'],
        reward_embedding_size=config['model']['reward_embedding_size'],
        rnn_hidden_size=config['model']['rnn_hidden_size'],
        dqn_layers=config['model']['dqn_layers'],
        policy_layers=config['model']['policy_layers'],
        lr=config['model']['lr'],
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
    ).to(ptu.device)

    # Initialize Wandb
    if config['wandb']['log']:
        wandb.init(project=config['wandb']['project_name'], entity=config['wandb']['entity'], config=config)

    # Initialize Trainer
    trainer = Trainer(config, env, obs_dim, action_dim, agent)

    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()

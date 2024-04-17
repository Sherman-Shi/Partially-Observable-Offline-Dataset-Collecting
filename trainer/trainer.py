import numpy as np
import torch
import wandb
import os 

from .buffers.seq_replay_buffer_vanilla import SeqReplayBuffer
from utils import helpers as utl
import torchkit.pytorch_utils as ptu


class Trainer:
    def __init__(self, config, env, agent):
        self.config = config
        self.env = env
        self.agent = agent
        self.policy_storage = SeqReplayBuffer(
            max_replay_buffer_size=int(self.config['training']['buffer_size']),
            observation_dim=self.config['model']['obs_dim'],
            action_dim=self.config['model']['action_dim'],
            sampled_seq_len=self.config['training']['sampled_seq_len'],
            sample_weight_baseline=0.0,
        )
        self.total_env_steps = 0
        self.save_interval = config['training']['save_interval']

    def train(self):
        for iter_num in range(self.config['training']['num_iters']):
            # Collect initial exploration data
            if iter_num < self.config['training']['num_init_rollouts_pool']:
                self.collect_rollouts(self.config['training']['num_rollouts_per_iter'], random_actions=True)

            # Regular training rollouts
            _ = self.collect_rollouts(self.config['training']['num_rollouts_per_iter'], train_mode=True)

            # Update the model with the collected data
            loss_info = self.update()

            # Log training information using Wandb
            if self.config['wandb']['log']:
                wandb.log(loss_info, step=iter_num)
            print(f"Loss: iter {iter_num}, loss {loss_info}\n")

            # Evaluation and logging the performance
            if iter_num % self.config['training']['log_interval'] == 0:
                eval_rewards = self.collect_rollouts(self.config['training']['eval_num_rollouts'], deterministic=True, train_mode=False)
                if self.config['wandb']['log']:
                    wandb.log({'eval_rewards': eval_rewards}, step=iter_num)

            if iter_num % self.save_interval == 0 and iter_num > 0:
                self.save_models(iter_num)

            # Print training status
            print(f"Completed iteration {iter_num + 1}/{self.config['training']['num_iters']}, Loss: {loss_info}, Eval Reward: {eval_rewards}")


    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False, deterministic=False, train_mode=True):
        """
        Collect num_rollouts of trajectories in the task and save them into the policy buffer if in training mode,
        otherwise just return the average episodic rewards.

        Parameters:
        - num_rollouts: Number of rollout episodes to perform.
        - random_actions: Whether to use random actions instead of the policy's actions.
        - deterministic: Whether to select actions deterministically.
        - train_mode: Whether the rollouts are used for training (data saved to buffer) or testing (compute average reward).

        Returns:
        - Average episodic reward if not in train mode, total number of steps if in train mode.
        """
        total_steps = 0
        total_rewards = 0.0

        for _ in range(num_rollouts):
            obs = ptu.from_numpy(self.env.reset()).unsqueeze(0)
            done_rollout = False
            episodic_reward = 0.0

            # Initial action, reward, and internal state
            action, reward, internal_state = self.agent.get_initial_info()

            # Storage for training data
            if train_mode:
                obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor([self.env.action_space.sample()])
                else:
                    (action, _, _, _), internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        deterministic=deterministic
                    )

                next_obs, reward, done, info = utl.env_step(self.env, action.squeeze(dim=0))
                episodic_reward += reward.item()
                done_rollout = done or "TimeLimit.truncated" in info

                if train_mode:
                    obs_list.append(obs)
                    act_list.append(action)
                    rew_list.append(reward)
                    term_list.append(int(not done_rollout))
                    next_obs_list.append(next_obs)

                obs = next_obs.clone()
                #obs = next_obs.unsqueeze(0)

            # Aggregate data for this episode
            if train_mode:
                total_steps += len(obs_list)

            total_rewards += episodic_reward

            if train_mode:
                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),
                    actions=ptu.get_numpy(torch.cat(act_list, dim=0)),
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),
                    terminals=np.array(term_list).reshape(-1, 1),
                    next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0))
                )

        # Return average reward per episode if not in training mode, total steps if in training mode
        return total_rewards / num_rollouts if not train_mode else total_steps



    def update(self):
        batch = ptu.np_to_pytorch_batch(self.policy_storage.random_episodes(self.config['training']['batch_size']))
        loss_info = self.agent.update(batch)
        return loss_info

    def save_model(self, model, directory, filename):
        """
        Saves a single model's state dictionary to a specified file within a directory.
        Creates the directory if it does not exist.
        
        Parameters:
        - model: The PyTorch model whose state_dict is to be saved.
        - directory: The directory path where the model will be saved.
        - filename: The filename for saving the model.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def save_models(self, iter_num):
        """
        Saves the state dictionaries of the TD3 actor and RNN model based on the iteration number.
        Models are saved in a directory structured by environment, algorithm, and encoder used.
        
        Parameters:
        - iter_num: Current training iteration number, used to version the saved models.
        """
        directory = os.path.join(
            "saved_models",
            self.env.spec.id if hasattr(self.env, 'spec') else 'default_env',
            self.agent.algo.name,
            self.agent.encoder,
            f"train_iter_{iter_num}"
        )

        self.save_model(self.agent.actor.policy, directory, "td3_actor.pth")
        self.save_model(self.agent.actor.rnn, directory, "rnn_model.pth")
        print(f"Models saved for iteration {iter_num} in directory {directory}")


    def load_model(model, filepath):
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {filepath}")

    def load_models(self, directory="model_checkpoints"):
        """
        Loads the state dictionaries into the TD3 actor and RNN model.

        Parameters:
        - actor: The TD3 actor model.
        - rnn: The RNN model.
        - directory: Directory from which models will be loaded.
        """
        actor_filepath = os.path.join(directory, "td3_actor.pth")
        rnn_filepath = os.path.join(directory, "rnn_model.pth")
        self.load_model(self.agent.algo.actor, actor_filepath)
        self.load_model(self.agent.actor.rnn, rnn_filepath)

# Additional methods for collect_rollouts and update can be modeled based on the provided main script logic.

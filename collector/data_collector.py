import gymnasium as gym 
import numpy as np
import torch
import wandb

def from_numpy(np_array, device):
    return torch.tensor(np_array, device=device).float()  # Ensure the tensor is on the correct device

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

class DataCollector:
    def __init__(self, config,  model, max_path_length, num_data_points, random=False):
        self.model = model
        self.policy = model.load_policy()  # Load policy from the model
        self.env = model.env  # Use the environment from the model
        self.max_path_length = max_path_length
        self.num_data_points = num_data_points
        self.random = random
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_wandb = config['wandb']['load']

    def get_reset_data(self):
        return {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'timeouts': [],
            'logprobs': []
        }

    def collect_data(self):
        data = self.get_reset_data()
        traj_data = self.get_reset_data()

        _returns, t, done = 0, 0, False
        s = self.env.reset()[0]
        while len(data['rewards']) < self.num_data_points:

            torch_s = from_numpy(np.expand_dims(s, axis=0), self.device)
            action = self.policy.sample()
            logprob = self.policy.log_prob(action)

            # Convert action to a simple integer if the action space is Discrete
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                a = int(action.item())  # Make sure action is a Python int
            else:
                a = action.numpy().squeeze()

            # Mujoco-specific data
            #qpos, qvel = self.env.sim.data.qpos.ravel().copy(), self.env.sim.data.qvel.ravel().copy()
            ns, rew, done, _, _ = self.env.step(a)
            # try:
            #     ns, rew, done, _, _ = self.env.step(a)
            # except:
            #     print('Lost connection. Resetting environment.')
            #     self.env.close()
            #     self.env = gym.make(self.env.unwrapped.spec.id)
            #     s = self.env.reset()
            #     traj_data = self.get_reset_data()
            #     t = 0
            #     _returns = 0
            #     continue

            _returns += rew
            t += 1
            timeout = t == self.max_path_length
            terminal = done if not timeout else False

            traj_data['observations'].append(s)
            traj_data['actions'].append(a)
            traj_data['next_observations'].append(ns)
            traj_data['rewards'].append(rew)
            traj_data['terminals'].append(terminal)
            traj_data['timeouts'].append(timeout)
            traj_data['logprobs'].append(logprob)

            s = ns
            if terminal or timeout:
                for k in data:
                    data[k].extend(traj_data[k])
                if self.load_wandb == True:
                    wandb.log({'Trajectory Reward': _returns, 'Trajectory Length': t, 'Steps': len(data["rewards"])})
                    print({'Trajectory Reward': _returns, 'Trajectory Length': t, 'Steps': len(data["rewards"])})
                    print("\n")
                print(f'Finished trajectory. Len={t}, Returns={_returns}. Progress:{len(data["rewards"])}/{self.num_data_points}')
                s = self.env.reset()[0]
                t, _returns = 0, 0

        
                traj_data = self.get_reset_data()

        for k in data:
            data[k] = np.array(data[k]).astype(np.float32 if 'logprobs' not in k else np.float32)
        return data

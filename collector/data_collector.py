import gym
import numpy as np
import torch

def from_numpy(np_array, device):
    return torch.tensor(np_array, device=device).float()  # Ensure the tensor is on the correct device

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

class DataCollector:
    def __init__(self, policy, env_name, max_path_length, num_data_points, random=False):
        self.policy = policy
        self.env = gym.make(env_name)
        self.max_path_length = max_path_length
        self.num_data_points = num_data_points
        self.random = random
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Class-wide device

    def get_reset_data(self):
        return {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'timeouts': [],
            'logprobs': [],
            'qpos': [],
            'qvel': []
        }

    def collect_data(self):
        data = self.get_reset_data()
        traj_data = self.get_reset_data()

        _returns, t, done = 0, 0, False
        s = self.env.reset()
        while len(data['rewards']) < self.num_data_points:
            if self.random:
                a = self.env.action_space.sample()
                logprob = np.log(1.0 / np.prod(self.env.action_space.high - self.env.action_space.low))
            else:
                torch_s = from_numpy(np.expand_dims(s, axis=0), self.device)
                distr = self.policy.forward(torch_s)
                a = distr.sample()
                logprob = distr.log_prob(a)
                a = to_numpy(a).squeeze()  # Ensure the action is back to numpy for gym

            # Mujoco-specific data
            qpos, qvel = self.env.sim.data.qpos.ravel().copy(), self.env.sim.data.qvel.ravel().copy()

            try:
                ns, rew, done, _ = self.env.step(a)
            except:
                print('Lost connection. Resetting environment.')
                self.env.close()
                self.env = gym.make(self.env.unwrapped.spec.id)
                s = self.env.reset()
                traj_data = self.get_reset_data()
                t = 0
                _returns = 0
                continue

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
            traj_data['qpos'].append(qpos)
            traj_data['qvel'].append(qvel)

            s = ns
            if terminal or timeout:
                print(f'Finished trajectory. Len={t}, Returns={_returns}. Progress:{len(data["rewards"])}/{self.num_data_points}')
                s = self.env.reset()
                t, _returns = 0, 0
                for k in data:
                    data[k].extend(traj_data[k])
                traj_data = self.get_reset_data()

        for k in data:
            data[k] = np.array(data[k]).astype(np.float32 if 'logprobs' not in k else np.float32)
        return data

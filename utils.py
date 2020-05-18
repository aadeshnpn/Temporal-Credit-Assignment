"""Useful methods for PPO."""
import torch
import math
import imageio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd


class RLEnvironment(object):
    """An RL Environment, used for wrapping environments to run PPO on."""

    def __init__(self):
        super(RLEnvironment, self).__init__()

    def step(self, x):
        """Takes an action x, which is the same format as the output from a policy network.
        Returns observation (np.ndarray), reward (float), terminal (boolean)
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the environment.
        Returns observation (np.ndarray)
        """
        raise NotImplementedError()


class EnvironmentFactory(object):
    """Creates new environment objects"""

    def __init__(self):
        super(EnvironmentFactory, self).__init__()

    def new(self):
        raise NotImplementedError()


class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length


def multinomial_likelihood(dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)


def get_log_p(data, mu, sigma):
    """get negative log likelihood from normal distribution"""
    return -torch.log(torch.sqrt(2 * math.pi * sigma ** 2)) - (data - mu) ** 2 / (2 * sigma ** 2)


# def calculate_returns(trajectory, gamma):
#     current_return = 0
#     for i in reversed(range(len(trajectory))):
#         state, action_dist, action, reward = trajectory[i]
#         ret = reward + gamma * current_return
#         trajectory[i] = (state, action_dist, action, reward, ret)
#         current_return = ret


def calculate_returns(trajectory, gamma, finalrwd):
    ret = finalrwd
    for i in reversed(range(len(trajectory))):
        state, action_dist, action, rwd, s1 = trajectory[i]
        # print(i, state, action, rwd, s1)
        trajectory[i] = (state, action_dist, action, rwd, ret, s1)
        # print(i, ret, end=' ')
        ret = ret * gamma


def run_envs(env, embedding_net, policy, experience_queue, reward_queue,
                num_rollouts, max_episode_length, gamma, device):
    for _ in range(num_rollouts):
        current_rollout = []
        s = env.reset()
        episode_reward = 0
        for _ in range(max_episode_length):
            input_state = prepare_numpy(s, device)
            if embedding_net:
                input_state = embedding_net(input_state)

            action_dist, action = policy(input_state)
            action_dist, action = action_dist[0], action[0]  # Remove the batch dimension
            s_prime, r, t = env.step(action)
            # print(_, s_prime, r)
            if type(r) != float:
                print('run envs:', r, type(r))
            s1 = np.concatenate((s, action*1.0))
            current_rollout.append((s, action_dist.cpu().detach().numpy(), action, r, s1))
            episode_reward += r
            if t:
                break
            s = s_prime
        # print(current_rollout, gamma, episode_reward)
        calculate_returns(current_rollout, gamma, episode_reward)
        # print(current_rollout)
        experience_queue.put(current_rollout)
        reward_queue.put(episode_reward)


def prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)


def prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)


# def make_gif(rollout, filename):
#     with imageio.get_writer(filename, mode='I', duration=1 / 30) as writer:
#         for x in rollout:
#             writer.append_data((x[0][:, :, 0] * 255).astype(np.uint8)

class LossPlot:

    def __init__(
            self, directory, fname,
            title="Temporal-Credit-Assignment Performance",
            pname='temporal'):
        self.__dict__.update(locals())
        plt.style.use('fivethirtyeight')
        self.data = self.load_file()
        # print(dir(self.data))
        self.vloss = self.data['value_loss']
        self.ploss = self.data['policy_loss']
        self.avg_reward = self.data['avg_reward']
        self.data = [self.vloss, self.ploss, self.avg_reward]
        self.pname = '/' + pname

    def gen_plots(self):
        fig = plt.figure(figsize=[12, 20])
        color = ['orange', 'red', 'green']
        label = ['Value Loss', 'Policy Loss', 'Average Reward']
        ylabel = ['Loss', 'Loss', 'Avg Reward']
        plt.title(self.title)
        for i in range(0, 3):
            ax1 = fig.add_subplot(3, 1, i+1)
            ax1.plot(
                range(len(self.data[i])), self.data[i],
                color=color[i], label=label[i],
                linewidth=1.0)
            ax1.legend()
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel(ylabel[i])
            # ax1.set_yticks(np.arange(min(self.data[i]), max(self.data[i])+1, 10.0))
            ax1.set_yticks(np.linspace(min(self.data[i]), max(self.data[i])+1, 10))
            # ax1.set_title('Value Loss in Temporal Credit Assignment')
        plt.tight_layout()
        fig.savefig(
            '/tmp' + self.pname + '.pdf')  # pylint: disable = E1101
        fig.savefig(
            '/tmp' + self.pname + '.png')  # pylint: disable = E1101
        plt.close(fig)

    def load_file(self):
        data = pd.read_csv(
            self.directory + '/' + self.fname, sep=','  # pylint: disable=E1101
            , skipinitialspace=True)
        return data



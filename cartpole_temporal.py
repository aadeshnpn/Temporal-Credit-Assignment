"""Definition of PPO algorithm"""
import imageio
from itertools import chain
import math
from threading import Thread
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from queue import Queue
import gym
import pickle
import matplotlib.pyplot as plt
import pandas as pd


from utils import (
    run_envs, ExperienceDataset, prepare_tensor_batch,
    multinomial_likelihood, EnvironmentFactory, RLEnvironment,
    DataLoader, LossPlot
    )


class CartPoleEnvironmentFactory(EnvironmentFactory):
    def __init__(self):
        super(CartPoleEnvironmentFactory, self).__init__()

    def new(self):
        return CartPoleEnvironment()


class CartPoleEnvironment(RLEnvironment):
    def __init__(self):
        super(CartPoleEnvironment, self).__init__()
        self._env = gym.make('CartPole-v0')
        self.ereward = 0

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        s, r, t, _ = self._env.step(action.item())
        # print(s, r)
        self.ereward += r
        if t:
            return s, self.ereward, t
        else:
            return s, 0.0, t

    def reset(self):
        """Returns observation (np.ndarray)"""
        self.ereward = 0
        return self._env.reset()


class CartPolePolicyNetwork(nn.Module):
    """Policy Network for CartPole."""

    def __init__(self, state_dim=4, action_dim=2):
        super(CartPolePolicyNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, action_dim)
        )
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x, get_action=True):
        """Receives input x of shape [batch, state_dim].
        Outputs action distribution (categorical distribution) of shape [batch, action_dim],
        as well as a sampled action (optional).
        """
        scores = self._net(x)
        probs = self._softmax(scores)

        if not get_action:
            return probs

        batch_size = x.shape[0]
        actions = np.empty((batch_size, 1), dtype=np.uint8)
        probs_np = probs.cpu().detach().numpy()
        for i in range(batch_size):
            action_one_hot = np.random.multinomial(1, probs_np[i])
            action_idx = np.argmax(action_one_hot)
            actions[i, 0] = action_idx
        return probs, actions


## Original Value network for Cartpole Problem
class CartPoleValueNetwork(nn.Module):
    """Approximates the value of a particular CartPole state."""

    def __init__(self, state_dim=4):
        super(CartPoleValueNetwork, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(state_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        return self._net(x)


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.error = torch.nn.MSELoss()

    def forward(self, out, reward):
        # out = torch.sum(out)
        # out = out.view(1)
        # reward = reward.view(1)
        # reward = reward.double()    # .to('cuda:0')
        # out = out.double()  # .to('cuda:0')
        # print('loss', out.shape, reward.shape)
        return self.error(out, reward)


class Regression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(Regression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.view((1, *src.shape))
        output = self.transformer_encoder(src)
        return output


### Temporal reward prediction value network
### Implemented from paper Temporal Credit assignment
class ValueNetwork(nn.Module):
    """Approximates the value of a particular state."""

    def __init__(
            self, transformer, selfatten, regression):
        super(ValueNetwork, self).__init__()
        self.transformer = transformer
        self.regression = regression
        self.selfatten = selfatten

    def forward(self, x):
        """Receives an observation of shape [batch, state_dim].
        Returns the value of each state, in shape [batch, 1]
        """
        # return self._net(x)
        x = x.squeeze()
        hidden = self.transformer(x).squeeze()
        # print('hidden shape',hidden.shape)
        hidden = hidden.transpose(1, 0)
        z = torch.sigmoid(self.selfatten(hidden))
        # hidden = self.selfatten(hidden)
        # print(hidden.shape, z.shape)
        hidden_sum = hidden * z
        hidden_sum = torch.reshape(
            hidden_sum, (hidden_sum.shape[1], hidden_sum.shape[0]))
        # print(hidden_sum.shape)
        out = self.regression(hidden_sum)
        return out


def ppo(env_factory, policy, value, likelihood_fn, embedding_net=None, epochs=1000,
        rollouts_per_epoch=100, max_episode_length=200, gamma=0.99, policy_epochs=5,
        batch_size=256, epsilon=0.2, environment_threads=4, data_loader_threads=4,
        device=torch.device('cpu'), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
        gif_name='', gif_epochs=0, csv_file='latest_run.csv', valueloss= nn.MSELoss()):

    # Clear the csv file
    with open(csv_file, 'w') as f:
        f.write('avg_reward, value_loss, policy_loss\n')

    # Move networks to the correct device
    policy = policy.to(device)
    value = value.to(device)

    # Collect parameters
    params = chain(policy.parameters(), value.parameters())
    if embedding_net:
        embedding_net = embedding_net.to(device)
        params = chain(params, embedding_net.parameters())

    # Set up optimization
    # optimizer = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    optimizer = optim.Adam(params, lr=lr)
    value_criteria = valueloss

    # Calculate the upper and lower bound for PPO
    ppo_lower_bound = 1 - epsilon
    ppo_upper_bound = 1 + epsilon

    loop = tqdm(total=epochs, position=0, leave=False)

    # Prepare the environments
    environments = [env_factory.new() for _ in range(environment_threads)]
    rollouts_per_thread = rollouts_per_epoch // environment_threads
    remainder = rollouts_per_epoch % environment_threads
    rollout_nums = ([rollouts_per_thread + 1] * remainder) + ([rollouts_per_thread] * (environment_threads - remainder))

    for e in range(epochs):
        # Run the environments
        policy = policy.to('cpu')
        experience_queue = Queue()
        reward_queue = Queue()
        threads = [Thread(target=run_envs, args=(environments[i],
                                                  embedding_net,
                                                  policy,
                                                  experience_queue,
                                                  reward_queue,
                                                  rollout_nums[i],
                                                  max_episode_length,
                                                  gamma,
                                                  'cpu')) for i in range(environment_threads)]
        for x in threads:
            x.start()
        for x in threads:
            x.join()

        # Collect the experience
        rollouts = list(experience_queue.queue)
        avg_r = sum(reward_queue.queue) / reward_queue.qsize()
        loop.set_description('avg reward: % 6.2f' % (avg_r))
        policy = policy.to(device)
        # Make gifs
        # if gif_epochs and e % gif_epochs == 0:
        #     make_gif(rollouts[0], gif_name + '%d.gif' % e)

        # Update the policy
        experience_dataset = ExperienceDataset(rollouts)
        data_loader = DataLoader(experience_dataset, num_workers=data_loader_threads, batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)
        avg_policy_loss = 0
        avg_val_loss = 0
        for _ in range(policy_epochs):
            avg_policy_loss = 0
            avg_val_loss = 0
            for state, old_action_dist, old_action, reward, ret, s1 in data_loader:
                state = prepare_tensor_batch(state, device)
                old_action_dist = prepare_tensor_batch(old_action_dist, device)
                old_action = prepare_tensor_batch(old_action, device)
                ret = prepare_tensor_batch(ret, device).unsqueeze(1)
                s1 = prepare_tensor_batch(s1, device)
                optimizer.zero_grad()
                if state.shape[0] != 64:
                    continue
                # If there is an embedding net, carry out the embedding
                if embedding_net:
                    state = embedding_net(state)

                # Calculate the ratio term
                current_action_dist = policy(state, False)
                # print(current_action_dist.shape)
                current_likelihood = likelihood_fn(current_action_dist, old_action)
                old_likelihood = likelihood_fn(old_action_dist, old_action)
                ratio = (current_likelihood / old_likelihood)

                # Calculate the value loss
                # print(s1.shape)
                expected_returns = value(s1)
                # print(expected_returns.shape, ret.shape)
                val_loss = value_criteria(expected_returns, ret)
                # val_loss = value_criteria(expected_returns, reward.sum().detach())

                # Calculate the policy loss
                advantage = ret - expected_returns.detach()
                # print(ratio.shape, advantage.shape)
                lhs = ratio * advantage
                rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
                policy_loss = -torch.mean(torch.min(lhs, rhs))

                # For logging
                avg_val_loss += val_loss.item()
                avg_policy_loss += policy_loss.item()

                # Backpropagate
                loss = policy_loss + val_loss
                loss.backward()
                optimizer.step()

            # Log info
            avg_val_loss /= len(data_loader)
            avg_policy_loss /= len(data_loader)
            loop.set_description(
                'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f' % (avg_r, avg_val_loss, avg_policy_loss))
        with open(csv_file, 'a+') as f:
            f.write('%6.2f, %6.2f, %6.2f\n' % (avg_r, avg_val_loss, avg_policy_loss))
        print()
        loop.update(1)


def main(i):
    factory = CartPoleEnvironmentFactory()
    policy = CartPolePolicyNetwork()
    # value = CartPoleValueNetwork()
    transformer = TransformerModel(500, 5, 1, 200, 2)
    # transformer = transformer.to(device)
    selfatt = Attention(64, 5)
    # selfatt = selfatt.to(device)
    lregression = Regression(5, 1)
    # lregression = lregression.to(device)
    value = ValueNetwork(transformer, selfatt, lregression)

    ppo(factory, policy, value, multinomial_likelihood, epochs=30,
        rollouts_per_epoch=128, max_episode_length=200,
        gamma=1.0, policy_epochs=5, batch_size=64,
        device='cpu', valueloss=RegressionLoss(), csv_file='latest_run'+str(i)+'.csv')

    # draw_losses()

class RewardPlot:

    def __init__(
            self, directory, fname,
            title="Temporal-Credit-Assignment Performance",
            pname='temporal'):
        self.__dict__.update(locals())
        plt.style.use('fivethirtyeight')
        self.datas = []
        for i in range(20):
            self.data = self.load_file(i)
            # print(dir(self.data))
            self.vloss = self.data['value_loss']
            self.ploss = self.data['policy_loss']
            self.avg_reward = self.data['avg_reward']
            self.data = [self.vloss, self.ploss, self.avg_reward]
            self.datas.append(self.data)
        self.pname = '/' + pname

    def gen_plots(self):
        # fig = plt.figure(figsize=[12, 20])
        fig = plt.figure()

        # Save the data in pickel
        with open('data.pkl','wb') as f:
            pickle.dump(self.datas, f, -1)

        color = ['orange', 'red', 'green']
        dcolor = ['coral', 'orangered', 'lime']
        label = ['Value Loss', 'Policy Loss', 'Average Reward']
        ylabel = ['Value Loss', 'Policy Loss', 'Avg Reward']
        plt.title(self.title)
        self.datas = np.array(self.datas)
        print(self.datas.shape)
        for i in range(0, 3):
            if i !=2:
                continue
            data = self.datas[:,i,:]
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            xvalues = range(1, len(mean) + 1)
            field_max = mean + std
            field_min = mean - std
            # ax1 = fig.add_subplot(3, 1, i+1)
            ax1 = fig.add_subplot(1, 1, 1)
            # ax1.plot(
            #     range(len(self.data[i])), self.data[i],
            #     color=color[i], label=label[i],
            #     linewidth=1.0)
            # Plotting mean and standard deviation
            ax1.plot(
                xvalues, mean, color=color[i], label=label[i],
                linewidth=1.0)
            ax1.fill_between(
                xvalues, field_max, field_min, color=dcolor[i], alpha=0.3)
            ax1.legend()
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel(ylabel[i])
            # ax1.set_yticks(np.arange(min(self.data[i]), max(self.data[i])+1, 10.0))
            # ax1.set_yticks(np.linspace(min(mean), max(mean)+1, 10))
            # ax1.set_title('Value Loss in Temporal Credit Assignment')
        plt.tight_layout()
        fig.savefig(
            '/tmp' + self.pname + '.pdf')  # pylint: disable = E1101
        fig.savefig(
            '/tmp' + self.pname + '.png')  # pylint: disable = E1101
        plt.close(fig)

    def load_file(self, i):
        fname = self.fname + str(i) +'.csv'
        data = pd.read_csv(
            self.directory + '/' + fname, sep=','  # pylint: disable=E1101
            , skipinitialspace=True)
        return data


def draw_losses():
    fname = 'latest_run.csv'
    import os
    folder = os.getcwd()
    graph = LossPlot(folder, fname)
    graph.gen_plots()


def draw_variance_graph():
    fname = 'latest_run'
    import os
    folder = os.getcwd()
    graph = RewardPlot(folder, fname)
    graph.gen_plots()


if __name__ == '__main__':
    for i in range(30):
       main(i)
    # draw_losses()
    # draw_variance_graph()

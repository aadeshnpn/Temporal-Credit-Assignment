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

from utils import (
    run_envs, ExperienceDataset, prepare_tensor_batch,
    multinomial_likelihood, EnvironmentFactory, RLEnvironment,
    DataLoader
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

    def step(self, action):
        """action is type np.ndarray of shape [1] and type np.uint8.
        Returns observation (np.ndarray), r (float), t (boolean)
        """
        s, r, t, _ = self._env.step(action.item())
        return s, r, t

    def reset(self):
        """Returns observation (np.ndarray)"""
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


def ppo(env_factory, policy, value, likelihood_fn, embedding_net=None, epochs=1000,
        rollouts_per_epoch=100, max_episode_length=200, gamma=0.99, policy_epochs=5,
        batch_size=256, epsilon=0.2, environment_threads=1, data_loader_threads=1,
        device=torch.device('cpu'), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01,
        gif_name='', gif_epochs=0, csv_file='latest_run.csv'):

    # Clear the csv file
    with open(csv_file, 'w') as f:
        f.write('avg_reward, value_loss, policy_loss')

    # Move networks to the correct device
    policy = policy.to(device)
    value = value.to(device)

    # Collect parameters
    params = chain(policy.parameters(), value.parameters())
    if embedding_net:
        embedding_net = embedding_net.to(device)
        params = chain(params, embedding_net.parameters())

    # Set up optimization
    optimizer = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    value_criteria = nn.MSELoss()

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
                                                  device)) for i in range(environment_threads)]
        for x in threads:
            x.start()
        for x in threads:
            x.join()

        # Collect the experience
        rollouts = list(experience_queue.queue)
        avg_r = sum(reward_queue.queue) / reward_queue.qsize()
        loop.set_description('avg reward: % 6.2f' % (avg_r))

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
            for state, old_action_dist, old_action, reward, ret in data_loader:
                state = prepare_tensor_batch(state, device)
                old_action_dist = prepare_tensor_batch(old_action_dist, device)
                old_action = prepare_tensor_batch(old_action, device)
                ret = prepare_tensor_batch(ret, device).unsqueeze(1)

                optimizer.zero_grad()

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
                expected_returns = value(state)
                val_loss = value_criteria(expected_returns, ret)

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


def main():
    factory = CartPoleEnvironmentFactory()
    policy = CartPolePolicyNetwork()
    value = CartPoleValueNetwork()
    ppo(factory, policy, value, multinomial_likelihood, epochs=1000,
        rollouts_per_epoch=100, max_episode_length=200,
        gamma=0.99, policy_epochs=5, batch_size=256, device='cuda:0')


if __name__ == '__main__':
    main()

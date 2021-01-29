import torch
import torch.nn as nn
from torch.nn import functional as F
import gym
import random
import torch.optim as optim
from garage.envs import normalize, GymEnv
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.np.exploration_policies import EpsilonGreedyPolicy, AddOrnsteinUhlenbeckNoise
import numpy as np
from garage.envs import normalize, GymEnv
from collections import namedtuple
import sys
import copy
import pandas as pd
import math

torch.set_default_tensor_type('torch.FloatTensor')

#name = 'Hopper-v2'
#name = 'InvertedDoublePendulum-v2'    #[300, 400]
#name = 'Pendulum-v0'
name = 'HalfCheetah-v3'             #[64, 32]
#name = 'InvertedPendulum-v2'
#name = 'Ant-v2'
#name = 'Humanoid-v2'


env = GymEnv(name)
env = normalize(env)
policy_path = '/home/niroantonio/PycharmProjects/pythonProject5/data/local/experiment/' + name + '-policy.pt'
qf_path = '/home/niroantonio/PycharmProjects/pythonProject5/data/local/experiment/' + name + '-qf.pt'

path = '/home/niroantonio/PycharmProjects/pythonProject5/data/local/experiment/' + name + '.csv'

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_obs', 'reward', 'done')
)


class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_outputs)


    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x


class QF(nn.Module):

    def __init__(self, obs_space, num_actions):
        super(QF, self).__init__()
        self.fc1 = nn.Linear(obs_space, 64 + num_actions)
        self.fc2 = nn.Linear(64 + num_actions + 1, 32)
        self.out = nn.Linear(32, num_actions)

    def forward(self, state, action):
        #state_action = torch.cat([state, action], 1)
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], 1)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size):
        self.list = []
        self.dummy_list = []
        self.count = 0
        self.max_size = max_size

    def add_experience(self, observation, action, reward, next_observation, done):
        experience = Experience(observation, action,  next_observation, reward, done)
        if len(self.list) < self.max_size:
            self.list.append(experience)
        else:
            self.list[self.count] = experience
            self.count = (self.count + 1) % self.max_size

    def add_dummy_experience(self, observation, action, reward, next_observation, done):
        experience = Experience(observation, action,  next_observation, reward, done)
        if len(self.dummy_list) < self.max_size:
            self.dummy_list.append(experience)

    def extract_batch(self, batch_size):
        batch = random.sample(self.list, int(batch_size/2))
        dummy_batch = random.sample(self.dummy_list, int(batch_size/2))
        batch += dummy_batch
        return batch


class DDPGagent:
    def __init__(self, qf1, qf2, policy):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tau = 0.01
        self.discount = 0.99
        self.memory_size = int(1e6)
        self.batch_size = 128
        self.num_epochs = 2000
        self.step_per_epochs = 1000
        self.policy = policy
        self._qf1 = qf1
        self._qf2 = qf2

        self._target_policy = copy.deepcopy(self.policy)
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = optim.Adam(params=self.policy.parameters(), lr=1e-4)
        self._qf1_optimizer = optim.Adam(params=self._qf1.parameters(), lr=1e-3)
        self._qf2_optimizer = optim.Adam(params=self._qf2.parameters(), lr=1e-3)

        self.memory = ReplayBuffer(max_size=self.memory_size)

        self.strategy = AddOrnsteinUhlenbeckNoise(env_spec=env,
                                                  policy=self.policy,
                                                  sigma=0.2)

        obs, _ = env.reset()
        while len(self.memory.dummy_list) < self.memory_size:
            action = np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)
            step = env.step(action)
            next_obs = step.observation
            reward = step.reward
            done = step.last
            self.memory.add_dummy_experience(obs, action, reward, next_obs, done)
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs
            self.memory.list = self.memory.dummy_list

    def update_target(self):
        for t_param, param in zip(self._target_qf1.parameters(),
                                  self._qf1.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)

        for t_param, param in zip(self._target_qf2.parameters(),
                                  self._qf2.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)

        for t_param, param in zip(self._target_policy.parameters(),
                                  self.policy.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)

    def train_once(self, batch, gs):

        states = []
        actions = []
        next_states = []
        rews = []
        dones = []

        for exp in batch:
            states.append(torch.FloatTensor(exp.state).unsqueeze(0))
            actions.append(torch.tensor([exp.action], dtype=float).unsqueeze(0))
            next_states.append(torch.FloatTensor(exp.next_obs).unsqueeze(0))
            rews.append(torch.tensor([exp.reward], dtype=float).unsqueeze(0))
            dones.append(torch.tensor([exp.done], dtype=float).unsqueeze(0))

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.cat(actions)
        actions = actions.reshape(self.batch_size, env.action_space.shape[0])
        rews = torch.cat(rews)
        dones = torch.cat(dones)

        inputs = states
        next_inputs = next_states

        qval1 = self._qf1(inputs, actions.float())
        qval2 = self._qf2(inputs, actions.float())
        with torch.no_grad():
            next_actions = self._target_policy(next_inputs)
            noise = torch.tensor(np.random.uniform(0, 0.1, next_actions.shape), dtype=float)
            next_actions += noise
            target_qvals1 = self._target_qf1(next_inputs, next_actions)
            target_qvals2 = self._target_qf2(next_inputs, next_actions)

        target_qvals = []
        for t1, t2 in zip(target_qvals1, target_qvals2):
            if t1 < t2:
                target_qvals.append(t1)
            else:
                target_qvals.append(t2)
        target_qvals = torch.cat(target_qvals).unsqueeze(1)

        y = rews + (1.0 - dones) * self.discount * target_qvals

        # optimize qf
        qf_loss = torch.nn.MSELoss()
        qval_loss = qf_loss(qval1, y.float())
        self._qf1_optimizer.zero_grad()
        qval_loss.backward()
        self._qf1_optimizer.step()

        qval_loss = qf_loss(qval2, y.float())
        self._qf2_optimizer.zero_grad()
        qval_loss.backward()
        self._qf2_optimizer.step()

        pred_actions = self.policy(inputs)
        action_loss = -1 * self._qf1(inputs, pred_actions).mean()

        if gs % 2 == 0:
            # optimize policy
            self._policy_optimizer.zero_grad()
            action_loss.backward()
            self._policy_optimizer.step()
            self.update_target()


    def train(self):

        best_reward = 0
        episodes = []
        tot_rewards = []
        tot_steps = []
        global_steps = 0
        count = 0
        for epoch in range(self.num_epochs):
            obs, _ = env.reset()
            ep_reward = 0
            for step in range(self.step_per_epochs):
                #env.render(mode='human')
                if epoch < 100:
                    action, _ = self.policy.get_action(obs)
                    noise = np.clip(self.strategy._simulate(), env.action_space.low, env.action_space.high)
                    action += noise
                    step = env.step(action)
                else:
                    #action = self.policy(torch.FloatTensor(obs).detach().numpy())
                    action, _ = self.policy.get_action(obs)
                    step = env.step(action)

                next_obs = step.observation
                reward = step.reward
                done = step.last
                self.memory.add_experience(obs, action, reward, next_obs, done)

                if len(self.memory.list) > self.batch_size:
                    batch = self.memory.extract_batch(self.batch_size)
                    self.train_once(batch, global_steps)

                ep_reward += reward
                obs = next_obs
                global_steps += 1

                if done or step == self.step_per_epochs - 1:
                    episodes.append(epoch)
                    tot_rewards.append(ep_reward)
                    tot_steps.append(step)
                    sys.stdout.write(
                        "episode: {}, reward: {}\n".format(epoch, ep_reward))
                    df = pd.DataFrame({'Episode': episodes, 'Reward': tot_rewards, 'Duration': tot_steps})
                    df.to_csv(path, encoding='utf-8')
                    break
            if ep_reward >= 9000:
                count += 1
                print(count)
            else:
                count = 0
            # if count == 100:
            #     print('Environment solved')
            #     break
            if ep_reward > best_reward:
                best_policy_param = self.policy.state_dict()
                best_qf_param = self._qf1.state_dict()

        torch.save(best_policy_param, policy_path)
        torch.save(best_qf_param, qf_path)


def main():
    # policy = Policy(num_inputs=env.observation_space.shape[0], num_outputs=env.action_space.shape[0])
    # qf1 = QF(obs_space=env.observation_space.shape[0], num_actions=env.action_space.shape[0])
    # qf2 = QF(obs_space=env.observation_space.shape[0], num_actions=env.action_space.shape[0])

    qf1 = ContinuousMLPQFunction(env_spec=env,
                                hidden_sizes=[64, 32],
                                hidden_nonlinearity=F.relu)
    qf2 = ContinuousMLPQFunction(env_spec=env,
                                hidden_sizes=[64, 32],
                                hidden_nonlinearity=F.relu)

    policy = DeterministicMLPPolicy(env_spec=env,
                                hidden_sizes=[64, 32],
                                hidden_nonlinearity=F.relu,
                                output_nonlinearity=torch.tanh)

    agent = DDPGagent(qf1, qf2, policy)
    agent.train()


main()


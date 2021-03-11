import torch
from torch.nn import functional as F
import gym
from garage.torch.policies import DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


name = 'HalfCheetah-v3'
# name = 'InvertedPendulum-v2'
#name = 'InvertedDoublePendulum-v2'
#name = 'Hopper-v2'
name = "r2d2-v0"
policy_path = '/home/niroantonio/PycharmProjects/pythonProject5/data/local/experiment/' + name + '-policy4.pt'
env = gym.make(name)

policy = DeterministicMLPPolicy(env_spec=env,
                                hidden_sizes=[400, 300],
                                hidden_nonlinearity=F.relu,
                                output_nonlinearity=torch.tanh)

policy_dict = torch.load(policy_path)

policy.load_state_dict(policy_dict)
done = False
obs = env.reset()
print(obs)
while not done:
    action = policy(torch.FloatTensor(obs))
    next_obs, reward, done, _ = env.step(action.detach().numpy())
    env.render(mode='human')
    obs = next_obs

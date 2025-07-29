import numpy as np
import torch

from network import NeuralNet
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam

class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = NeuralNet(self.obs_dim, self.act_dim)
        self.critic = NeuralNet(self.obs_dim, 1)

        self._init_hyperparameters()

        self.cov_mat = torch.full(size=(self.act_dim,), fill_value=0.5)
        print(self.cov_mat)
        self.cov_mat = torch.diag(self.cov_mat)
        print(self.cov_mat)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 10000
        self.max_timesteps_per_episode = 1500
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_return = []
        batch_lengths = []

        t = 0
        
        while t < self.timesteps_per_batch:
            ep_rewards = []
            obs, _ = self.env.reset()
            done = False
    
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                #print(type(action))
                #if action[0] > 0: action[0] = 1
                #else: action[0] = 0
                #print(action)
                obs, reward, done, _, trunc = self.env.step(action)

                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                    
                if done: break
            
            batch_lengths.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_return = self.compute_return(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_return, batch_lengths

    def get_action(self, obs):
        mean = self.actor.forward(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def compute_return(self, batch_rewards):
        batch_return = []

        for ep_rewards in reversed(batch_rewards):
            discounted = 0

            for reward in reversed(ep_rewards):
                discounted = reward + discounted*self.gamma
                batch_return.insert(0, discounted)
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def learn(self, total_timesteps):
        current_step = 0
        
        while current_step < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_return, batch_lengths = self.rollout()
            current_step += np.sum(batch_lengths)

            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_return - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_return)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

    def demo(self, env):
        obs, _ = env.reset()
        for _ in range(1000):
            action, _ = self.get_action(obs)
            #action = np.argmax(action)
            #if action[0] > 0: action[0] = 1
            #else: action[0] = 0
            obs, r, d, t, i = e.step(action)
            if d: e.reset()
        e.close()

    def evaluate(self, batch_obs, batch_acts):
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return self.critic(batch_obs).squeeze(), log_probs


if __name__ == "__main__":
    game = "Pendulum-v1"
#    game = "CartPole-v1"
    import gymnasium as gym
    env = gym.make(game)
    model = PPO(env)
    model.learn(100000)
    e = gym.make(game, render_mode="human")
    model.demo(e)

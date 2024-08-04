import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


# Define the ActorCritic class
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def pi(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc_pi(x)
        return Categorical(logits=x)
    
    def v(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc_v(x)
        return v
    

# Define the PPO class
class PPOAgent:
    def __init__(self, state_dim, action_dim, buffer_size, gamma,
                  K_epochs, eps_clip,hidden_dim=128,device=None):

        self.policy = ActorCritic(state_dim, action_dim,hidden_dim)
        self.policy_old = ActorCritic(state_dim, action_dim,hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = self.policy.optimizer
        self.MseLoss = nn.MSELoss()
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.rewards = []

    def update(self):
        # states = []
        # actions = []
        # logprobs = []
        # rewards = []
        # is_terminals = []
        states, actions, logprobs, rewards, is_terminals = zip(*self.memory)


        # for (state, action, logprob, reward, is_terminal) in self.memory:
        #     states.append(state)
        #     actions.append(action)
        #     logprobs.append(logprob)
        #     rewards.append(reward)
        #     is_terminals.append(is_terminal)

        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(states).detach())
        old_actions = torch.squeeze(torch.stack(actions).detach())
        old_logprobs = torch.squeeze(torch.stack(logprobs).detach())

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, discounted_rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate(self, state, action):
        state_value = self.policy.v(state)
        dist = self.policy.pi(state)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save(self,filename):
        torch.save(self.policy.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self,filename):
        self.policy.load_state_dict(torch.load(filename,map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"Model loaded from {filename}")

    def normalize_state(self, state):
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def train(self, env, num_episodes, early_stopping=None, checkpoint_path=None):
        for episode in range(1, num_episodes + 1):
            total_reward = 0
            state = env.reset()
            state = self.normalize_state(state)
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                dist = self.policy_old.pi(state_tensor)
                action = dist.sample()

                next_state, reward, done, _ = env.step(action.item())
                next_state = self.normalize_state(next_state)

                self.memory.append((state_tensor, action, dist.log_prob(action), reward, done))

                state = next_state

                total_reward += reward
                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break

            self.update()
            self.memory.clear()
            self.rewards.append(total_reward)

            if early_stopping and early_stopping(self.rewards):
                print("Early stopping criterion met")
                if checkpoint_path:
                    self.save(checkpoint_path)
                break

        env.close()

    

    def test(self, env, num_episodes=10):
        for episode in range(num_episodes):
            state = env.reset()
            state = self.normalize_state(state)
            done = False
            total_reward = 0
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                dist = self.policy_old.pi(state_tensor)
                action = dist.sample()
                state, reward, done, _ = env.step(action.item())
                state = self.normalize_state(state)
                total_reward += reward
                # env.render()
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            self.rewards.append(total_reward)
        env.close()


    def plot(self,plot_path='ppo.png'):
                
        data = self.rewards

        # Calculate the moving average
        window_size = 10
        moving_avg = pd.Series(data).rolling(window=window_size).mean()

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot the moving average line
        sns.lineplot(data=moving_avg, color='red')

        # Shade the area around the moving average line to represent the range of values
        plt.fill_between(range(len(moving_avg)),
                        moving_avg - np.std(data),
                        moving_avg + np.std(data),
                        color='blue', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Moving Average of Rewards')
        plt.grid(True)
        # Adjust layout to prevent overlapping elements
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(plot_path)
        # Show the plot
        # plt.show()

        return plt






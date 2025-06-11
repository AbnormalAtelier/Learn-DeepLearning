import numpy as np
import matplotlib.pyplot as plt
import subprocess

class MultiArmedBandit:
    def __init__(self, k):
        self.probs = [0.00793833, 0.35648651, 0.45312573, 0.18244944] #np.random.dirichlet(np.ones(k))
    
    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0
    
class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.1, k=2):
        self.epsilon = epsilon
        self.counts = np.zeros(k, dtype=int)
        self.values = np.zeros(k)
        self.k = k

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.k, dtype=int))
        else: 
            return int(np.argmax(self.values))
        
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value)/n

def run_simulation(bandit, agent, steps =1_000):
    rewards = []
    for _ in range(steps):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        rewards.append(reward)
    return rewards


bandit = MultiArmedBandit(4)
agent = EpsilonGreedyAgent(epsilon=0.175)
rewards = run_simulation(bandit, agent, steps=10_000)

cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

print(np.sum(rewards)/10_000)
print (bandit.probs)

plt.plot(cumulative_avg)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Performance")
plt.grid(True)
plt.savefig("e_greedy_reward_plot.png")
subprocess.Popen('xdg-open e_greedy_reward_plot.png', shell=True)

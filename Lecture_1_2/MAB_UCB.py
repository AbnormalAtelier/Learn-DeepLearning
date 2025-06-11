import numpy as np
import matplotlib.pyplot as plt
import subprocess

class MultiArmedBandit:
    def __init__(self, k=4):
        self.probs = [0.00793833, 0.35648651, 0.45312573, 0.18244944] #np.random.dirichlet(np.ones(k))
    
    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0
    
class UCBAgent:
    def __init__(self, k=4):
        self.counts = np.zeros(k, dtype=int)
        self.values = np.zeros(k)
        self.k = k

    def select_arm(self):
        total_counts = np.sum(self.counts)
        UCB_values = np.zeros(self.k)

        for i in range(self.k):
            if self.counts[i] == 0:
                UCB_values[i] = float('inf')  # force selection
            else:
                bonus = np.sqrt(2 * np.log(total_counts) / self.counts[i])
                UCB_values[i] = self.values[i] + bonus

        return np.argmax(UCB_values)

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
agent = UCBAgent()
rewards = run_simulation(bandit, agent, steps=1_000)

cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

print(np.sum(rewards)/1_000)
print (bandit.probs)

plt.plot(cumulative_avg)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB Performance")
plt.grid(True)
plt.savefig("e_greedy_reward_plot.png")
subprocess.Popen('xdg-open e_greedy_reward_plot.png', shell=True)
    
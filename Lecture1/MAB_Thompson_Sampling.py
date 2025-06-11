import numpy as np
import matplotlib.pyplot as plt
import subprocess

class MultiArmedBandit:
    def __init__(self, k=4):
        self.probs = [0.00793833, 0.35648651, 0.45312573, 0.18244944] #np.random.dirichlet(np.ones(k))
    
    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0
    
class ThompsonSamplingAgent:
    def __init__(self, k=4):
        self.k = k
        self.successes = np.zeros(k)
        self.failures = np.zeros(k)

    def select_arm(self):
        sampled_values = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(sampled_values)

    def update(self, arm, reward):
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

def run_simulation(bandit, agent, steps =1_000):
    rewards = []
    for _ in range(steps):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        rewards.append(reward)
    return rewards


bandit = MultiArmedBandit(4)
agent = ThompsonSamplingAgent()
rewards = run_simulation(bandit, agent, steps=1_000)

cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

print(np.sum(rewards)/1_000)
print (bandit.probs)

plt.plot(cumulative_avg)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Performance")
plt.grid(True)
plt.savefig("thompson_sampling_reward_plot.png")
subprocess.Popen('xdg-open thompson_sampling_reward_plot.png', shell=True)
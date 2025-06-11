import numpy as np
import matplotlib.pyplot as plt
import subprocess

class MultiArmedBandit:
    def __init__(self, k=4):
        self.probs = [0.00793833, 0.35648651, 0.45312573, 0.18244944] #np.random.dirichlet(np.ones(k))
    
    def pull(self, arm):
        return 1 if np.random.rand() < self.probs[arm] else 0
    
class PolicyGradientAgent:
    def __init__(self, k=4, alpha=0.1):
        self.k = k
        self.alpha = alpha
        self.preferences = np.zeros(k)  # h(a)
    
    def softmax(self):
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  # for numerical stability
        return exp_prefs / np.sum(exp_prefs)

    def select_arm(self):
        probs = self.softmax()
        return np.random.choice(self.k, p=probs)
    
    def update(self, arm, reward):
        probs = self.softmax()
        grad_log_pi = -probs
        grad_log_pi[arm] += 1  # ∇logπ(a) = 1 - p(a), -p(other)
        self.preferences += self.alpha * reward * grad_log_pi  # policy gradient update

def run_simulation(bandit, agent, steps =1_000):
    rewards = []
    for _ in range(steps):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        rewards.append(reward)
    return rewards

bandit = MultiArmedBandit(4)
agent = PolicyGradientAgent(k=4, alpha=0.1)
rewards = run_simulation(bandit, agent, steps=1_000)


cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

print(np.sum(rewards)/1_000)
print (bandit.probs)

plt.plot(cumulative_avg)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB Performance")
plt.grid(True)
plt.savefig("policy_gradient_reward_plot.png")
subprocess.Popen('xdg-open policy_gradient_reward_plot.png', shell=True)

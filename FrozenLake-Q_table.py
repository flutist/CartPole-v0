import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

s_space = env.observation_space.n
a_sapce = env.action_space.n

Q = np.zeros([s_space, a_sapce])
learning_rate = 0.85
reward_discount_rate = 0.99
total_episodes = 2000

rList = []
for episode in range(total_episodes):
    s = env.reset()
    total_reward = 0
    done = False

    for i in range(99):
        # generate random noise of action, decrease linearly
        random_action_noise = np.random.randn(1,env.action_space.n)*(1./(episode+1))
        # add random prob. in action and the q table
        a = np.argmax(Q[s,:] +  random_action_noise)

        s1, r, done, info = env.step(a)

        td_error =  (r + reward_discount_rate*np.max(Q[s1,:]) - Q[s,a])
        Q[s,a] = Q[s,a] + learning_rate * td_error

        total_reward += r
        s = s1
        if done:
            break
    rList.append(total_reward)

plt.plot(rList)
plt.title("Total reward over episode")
plt.ylabel('rewards')
plt.show()

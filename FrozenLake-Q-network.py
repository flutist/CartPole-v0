# -*- coding: utf-8 -*-
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


learning_rate = 0.1
reward_discount_rate = 0.99
total_episodes = 2000
epsilon = 0.1

env = gym.make("FrozenLake-v0")

s_space = env.observation_space.n
a_sapce = env.action_space.n

tf.reset_default_graph()

x = tf.placeholder(shape=[1, s_space],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([s_space,a_sapce], 0, 0.01))
Qout = tf.matmul(x,W)
predict = tf.argmax(Qout, 1)

Q_ = tf.placeholder(shape=[1,a_sapce],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(Q_ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

rList = []

with tf.Session() as sess:
    sess.run(init)
    for episode in range(total_episodes):
        s = env.reset()
        total_reward = 0
        done = False

        for i in range(99):
            a, allQ = sess.run([predict, Qout], feed_dict={
                    x: np.identity(16)[s:s+1]
                    })
            if np.random.rand(1) < epsilon:
                a[0] = env.action_space.sample() #np.random.randint(0, a_sapce-1)

            s1, r, done, info = env.step(a[0])


            allQ1 = sess.run([Qout], feed_dict={
                    x: np.identity(16)[s1:s1+1]
                    })
            maxQ1 = np.max(allQ1)
            #bootsrap target q value
            targetQ = allQ
            targetQ[0,a[0]] = r + reward_discount_rate * maxQ1
            _ = sess.run([updateModel], feed_dict={
                        x: np.identity(16)[s:s+1],
                        Q_: targetQ
                        })

            total_reward += r
            s = s1
            if done:
                epsilon = 1./((i/50) + 10)
                break
        rList.append(total_reward)

    plt.plot(rList)
    plt.title("Total reward over episode")
    plt.ylabel('rewards')
    plt.show()
    print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

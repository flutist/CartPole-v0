# This file is to implement A3C method
import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
import random, numpy, math, gym
import numpy as np


class Brian:
   def __init__(self, stateSize, actionSize, scope, trainer, sess):
       self.stateSize = stateSize
       self.actionSize = actionSize
       self.scope = scope
       self.trainer = trainer
       self.sess = sess

       self._createModel()

   def _createModel(self):
       with tf.variable_scope(self.scope):
           self.inputs = tf.placeholder('float', shape=[None,self.stateSize])
           x1 = slim.fully_connected(
               self.input,
               64,
               scope='fc/fc_1',
               activation_fn=tf.nn.relu)

           self.policy = slim.fully_connected(x1, self.actionSize,
               activation_fn=tf.nn.softmax,
               weights_initializer=Brian.normalized_columns_initializer(0.01),
               biases_initializer=None)
           self.value = slim.fully_connected(x1,1,
               activation_fn=None,
               weights_initializer=Brian.normalized_columns_initializer(1.0),
               biases_initializer=None)

           self.update_local_ops = Brian.update_target_graph('global',self.scope)

           if self.scope != 'global':
               self.actions = tf.placeholder('float', shape=[None])
               self.actions_onehot = tf.one_hot(self.actions, self.actionSize, dtype=tf.float32)
               self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
               self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

               self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

               #Loss functions
               self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
               self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
               self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
               self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

               #Get gradients from local network using local losses
               local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
               self.gradients = tf.gradients(self.loss,local_vars)
               self.var_norms = tf.global_norm(local_vars)
               grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

               #Apply local gradients to global network
               global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
               self.apply_grads = self.trainer.apply_gradients(zip(grads,global_vars))


   @classmethod
   def normalized_columns_initializer(std=1.0):
       def _initializer(shape, dtype=None, partition_info=None):
           out = np.random.randn(*shape).astype(np.float32)
           out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
           return tf.constant(out)
       return _initializer

   # Discounting function used to calculate discounted returns.
   @classmethod
   def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

   @classmethod
   def update_target_graph(from_scope,to_scope):
       from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
       to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

       op_holder = []
       for from_var,to_var in zip(from_vars,to_vars):
           op_holder.append(to_var.assign(from_var))
   return op_holder

   def train(self, states, action, target_v, advantages):
       feed_dict = {
           self.brian.inputs : states,
           self.brian.actions : action,
           self.brian.target_v : target_v,
           self.brian.advantages : advantages
       }
       return self.sess.run([
         self.value_loss,
         self.policy_loss,
         self.grad_norms,
         self.var_norms,
         self.apply_grads
       ], feed_dict=feed_dict)

   def predict(self, state):
       return self.sess.run([self.value, self.policy],
                            feed_dict={self.inputs: state})

   def update_ops(self):
       self.sess.run(self.update_local_ops)

   def predictOne(self, s):
       a_dist, v_dist = self.predict(s.reshape(1, self.stateSize))
       a = np.random.choice(a_dist[0],p=a_dist[0])
       a = np.argmax(a_dist == a)
       return a, v_dist[0]




class Agent:
    MAX_BUFFER_SIZE = 30
    def __init__(self, stateSize, actionSize, name, trainer, sess):
        self.name = name
        self.brian = Brian(stateSize, actionSize, self.name, trainer, sess)
        self.buffer = []

    def train(self, gamma=0.99, bootstrap_value):
        '''
        gamma is the dicount factor
        bootstrap_value is the reward of the final action, it can be bootstraped or a final actual value
        '''
        rollout = np.array(self.buffer)
        s  = rollout[:,0]
        a  = rollout[:,1]
        r  = rollout[:,2]
        s_ = rollout[:,3]
        v  = rollout[:,5]

        rewards_plus = np.asarray(r.tolist() + [bootstrap_value])
        discounted_rewards = Brian.discount(rewards_plus, gamma)
        value_plus = np.asarray(v.tolist() + [bootstrap_value])

        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = Brian.discount(advantages,gamma)

        r_l, p_l, e_l, g_n, v_n = self.brian.train(s, a, discounted_rewards, advantages)
        return r_l, p_l, e_l, g_n, v_n

    def update_ops(self):
        self.brian.update_ops()

    def observe(self, sample):
        self.buffer.append(sample)

    def act(self, state):
        return self.brian.predictOne()

    def isBufferFulled(self):
        return len(self.buffer) >= self.MAX_BUFFER_SIZE

    def episodeEnd(self):
        self.train()


class Environment:
    def __init__(self):
        self.env = gym.make(PROBLEM)

    def run(self, agent):
        s = self.env.reset()
        R = 0

        while True:
            a, v = agent.act(s)

            s_, r, done, info = self.env.step(a)

            agent.observe((s, a, r, s_, d, v))

            if agent.isBufferFulled() or done:
                agent.train()
                agent.update_ops()

            s = s_
            R += r

            if done:
                break
        agent.episodeEnd()

class Worker:
    '''
        Each worker will store it's own copy of hyper config
    '''
    def __init__(self, max_episode_length,
                 gamma, sess, coord, saver, global_episodes,
                 trainer, s_size, a_size, model_path, number):
        self.max_episode_length = max_episode_length
        self.gamma = gamma
        self.sess = sess
        self.coord = coord
        self.saver = saver
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.trainer = trainer
        self.s_size = s_size
        self.a_size = a_size
        self.model_path = model_path
        self.name = "worker_" +  str(number)
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.name))

        self.agent = Agent()
        self.env = Environment()


    def run(self):
        agent = self.agent

        self.episode_count = sess.run(self.global_episodes)
        print("Starting worker " + str(self.name))

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                self.env.run(agent)
                if self.episode_count % 5 == 0 and self.episode_count!=0:
                    self.log()
                if self.episode_count > TOTAL_EPISODE:
                    coord.request_stop()
                self.episode_count += 1

    def log(self):
        pass


stateSize = 4
actionSize = 2
GAMMA = 0.99
TOTAL_EPISODE = 500
PROBLEM = 'CartPole-v0'
MODEL_PATH = "./model"
max_episode_length = 200
NU_THREADS = multiprocessing.cpu_count()
NU_THREADS = 1

tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    masterBrian = Brian(stateSize,actionSize,'global',None, None)
    saver = tf.train.Saver(max_to_keep=5)


with tf.Session() as sess:
    coord = tf.train.Coordinator()

    threads = []        # the threads
    workers = []        # the worker object

    for i in range(NU_THREADS):
        worker = Worker(max_episode_length,GAMMA,sess,coord, saver, global_episodes, trainer, stateSize, actionSize, MODEL_PATH, i )
        workers.append(worker)
        work = lambda: worker.run()
        t = threading.Thread(target=(work))
        workers.append(work)
        threads.append(t)

    # init vars
    sess.run(tf.global_variables_initializer())

    for t in threads:
        t.start()
        sleep(0.5)
    coord.join(threads)

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
               self.inputs,
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
               self.actions = tf.placeholder( shape=[None], dtype=tf.int32)
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


   @staticmethod
   def normalized_columns_initializer(std=1.0):
       def _initializer(shape, dtype=None, partition_info=None):
           out = np.random.randn(*shape).astype(np.float32)
           out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
           return tf.constant(out)
       return _initializer

   # Discounting function used to calculate discounted returns.
   @staticmethod
   def discount(x, gamma):
       return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

   @staticmethod
   def update_target_graph(from_scope,to_scope):
       from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
       to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

       op_holder = []
       for from_var,to_var in zip(from_vars,to_vars):
           op_holder.append(to_var.assign(from_var))
       return op_holder

   def train(self, states, action, target_v, advantages):
       feed_dict = {
           self.inputs : states,
           self.actions : action,
           self.target_v : target_v,
           self.advantages : advantages
       }
       
       return self.sess.run([
         self.value_loss,
         self.policy_loss,
         self.grad_norms,
         self.var_norms,
         self.apply_grads
       ], feed_dict=feed_dict)

   def predict(self, state):
       return self.sess.run([self.policy, self.value],
                            feed_dict={self.inputs: state})

   def update_ops(self):
       self.sess.run(self.update_local_ops)

   def predictOne(self, s):
       a_dist, v_dist = self.predict(s.reshape(1, self.stateSize))
       a = np.random.choice(a_dist[0],p=a_dist[0])
       a = np.argmax(a_dist == a)
       return a, v_dist.flatten()[0]




class Agent:
    MAX_BUFFER_SIZE = 30
    def __init__(self, stateSize, actionSize, name, trainer, sess):
        self.name = name
        self.brian = Brian(stateSize, actionSize, self.name, trainer, sess)
        self.buffer = []

    def train(self, gamma, bootstrap_value):
        '''
        gamma is the dicount factor
        bootstrap_value is the reward of the final action, it can be bootstraped or a final actual value
        '''
        rollout = self.buffer
#        s  = rollout[:,0]
        s = np.array([ o[0] for o in rollout])
        a = np.array([ o[1] for o in rollout])
        
#        a  = rollout[:,1]
        r  = np.array([ o[2] for o in rollout])
        s_ = np.array([ o[3] for o in rollout])
        v  = np.array([ o[5] for o in rollout])

        rewards_plus = np.asarray(r.tolist() + [bootstrap_value])
        discounted_rewards = Brian.discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(v.tolist() + [bootstrap_value])

        advantages = r + gamma * value_plus[1:] - value_plus[:-1]
        advantages = Brian.discount(advantages,gamma)

        r_l, p_l, e_l, g_n, v_n = self.brian.train(s, a, discounted_rewards, advantages)
        
        #clear buffer
        self.buffer=[]
        return r_l, p_l, e_l, g_n, v_n, np.mean(advantages)

    def update_ops(self):
        self.brian.update_ops()

    def observe(self, sample):
        self.buffer.append(sample)

    def act(self, state):
        return self.brian.predictOne(state)

    def isBufferFulled(self):
        return len(self.buffer) >= self.MAX_BUFFER_SIZE

    def episodeEnd(self, gamma, bootstrap_value):
        if(len(self.buffer)==0):
            return None, None, None, None, None, None
        return self.train(gamma, bootstrap_value)


class Environment:
    def __init__(self):
        self.env = gym.make(PROBLEM)
        print("start "+PROBLEM)
        
        self.r_l = []
        self.p_l = []
        self.e_l = []
        self.g_n = []
        self.v_n = []
        self.mean_advantages = []
    
    def addLogBuffer(self, r_l, p_l, e_l, g_n, v_n, mean_advantages):        
        if v_n == None:
            v_n = 0
        self.r_l.append(r_l)
        self.p_l.append(p_l)
        self.e_l.append(e_l)
        self.g_n.append(g_n)
        self.v_n.append(v_n)
        self.mean_advantages.append(mean_advantages)
        
    def getMeanBuffer(self):
        return np.mean(self.r_l), np.mean(self.p_l), np.mean(self.e_l), np.mean(self.g_n), np.mean(self.v_n),   np.mean(self.mean_advantages)

    def run(self, agent):
        s = self.env.reset()
        R = 0
        
        self.r_l = []
        self.p_l = []
        self.e_l = []
        self.g_n = []
        self.v_n = []
        self.mean_advantages = []

        while True:
            a, v = agent.act(s)

            s_, r, done, info = self.env.step(a)

            agent.observe((s, a, r, s_, done, v))

            if agent.isBufferFulled() or done:
                a_, v_ = agent.act(s_)
                r_l, p_l, e_l, g_n, v_n, mean_advantages = agent.train(0.99, v_)
                self.addLogBuffer(r_l, p_l, e_l, g_n, v_n, mean_advantages)
                agent.update_ops()

            s = s_
            R += r

            if done:
                break
        r_l, p_l, e_l, g_n, v_n, mean_advantages = agent.episodeEnd(0.99, 0)
        if r_l != None:
            self.addLogBuffer(r_l, p_l, e_l, g_n, v_n, mean_advantages)
        r_l_m, p_l_m, e_l_m, g_n_m, v_n_m, mean_advantages_m = self.getMeanBuffer()
        agent.update_ops()
        return R, r_l_m, p_l_m, e_l_m, g_n_m, v_n_m, mean_advantages_m

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
        print("created worker ", number)
        print("id(self) ",id(self))
        self.name = "worker_" +  str(number)
        self.summary_writer = tf.summary.FileWriter("logs/train_"+str(self.name))

        self.agent = Agent(s_size, a_size, self.name, self.trainer, sess)
        self.env = Environment()


    def run(self):
        print("Starting worker " + str(self.name))
        print("Starting self.agent.name " + str(self.agent.name))
        print("id(self) ",id(self))
        agent = self.agent

        self.episode_count = sess.run(self.global_episodes)

        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                R, r_l, p_l, e_l, g_n, v_n, mean_advantages_m = self.env.run(agent)
                if self.episode_count % 5 == 0 and self.episode_count!=0:
                    self.log(R, r_l, p_l, e_l, g_n, v_n, mean_advantages_m)
                if self.episode_count > TOTAL_EPISODE:
                    coord.request_stop()
                self.episode_count += 1

    def log(self, rewards, v_l, p_l, e_l, g_n, v_n, mean_advantages_m):
        print(str(self.name), " episode_count", self.episode_count)
        summary = tf.Summary()
        summary.value.add(tag='Perf/Reward', simple_value=float(rewards))
#        summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
#        summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
        summary.value.add(tag='Losses/mean_advantages_m', simple_value=float(mean_advantages_m))
        self.summary_writer.add_summary(summary, self.episode_count)

        self.summary_writer.flush()
        pass


stateSize = 4
actionSize = 2
GAMMA = 0.99
TOTAL_EPISODE = 5000
PROBLEM = 'CartPole-v0'
MODEL_PATH = "./model"
max_episode_length = 200
NU_THREADS = multiprocessing.cpu_count()
#NU_THREADS = 2

SINGLE_THREAD = False

print("NU_THREADS " + str(NU_THREADS))

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
        print("create worker ", i)
        worker = Worker(max_episode_length,GAMMA,sess,coord, saver, global_episodes, trainer, stateSize, actionSize, MODEL_PATH, i )
        print("worker id(self) ",id(worker))
        print('-----------')
        workers.append(worker)

    # init vars
    sess.run(tf.global_variables_initializer())
    print("*******************************")

    if SINGLE_THREAD:
        workers[0].run()
    else:    
        for i in range(NU_THREADS):
            work = lambda: workers[i].run()
            t = threading.Thread(target=(work))
            t.start()
            threads.append(t)
            print("workers[i].name ",workers[i].name)
            print("threads[i] id(t) ",id(t))
            print('-----------')
            sleep(0.5)
        coord.join(threads)

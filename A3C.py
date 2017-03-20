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

class Brain:
    def __init__(self, stateSize, actionSize, scope, optimizer):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.scope = scope

        self._createModel()

    def _createModel(self):
        with tf.variable_scope(self.scope):
            self.inputs = tf.placeholder('float', shape=[None,self.stateSize])
            x1 = slim.fully_connected(
                self.input,
                64,
                scope='fc/fc_1',
                activation_fn=tf.nn.relu)

            self.policy = slim.fully_connected(x1,actionSize,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(x1,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            if self.scope != 'global':
                self.actions = tf.placeholder('float', shape=[None])
                self.actions_onehot = tf.one_hot(self.actions, self.actionSize, dtype=yf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)



class Agent:
    pass

class Environment:
    pass

class worker:
    def __init__(self):
        pass

    @staticmethod
    def update_target_graph(from, to):
        '''
            init. a gprah to copy the var from other
        '''
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def copyFromGlobalNetwork(self):
        sess.run(self.update_local_ops)

    def writeLog(self):
        ''' write logs  '''
        pass

    def train(self):
        pass

    def run(self):
        ''' run the worker '''

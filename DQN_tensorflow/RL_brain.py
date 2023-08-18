"""
Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=200,
            e_greedy_increment=0.0001,  #
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0)
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.1 if e_greedy_increment is not None else self.epsilon_max
        self.lr_decay_rate = 0.95
        self.lr_decay_step = 10000
        self.lr = tf.train.exponential_decay(
            self.learning_rate, self.global_step,
            self.lr_decay_step, self.lr_decay_rate, staircase=True
        )
        self.l_r = self.learning_rate
        self.gama = 3  ###拉格朗日乘子
        self.tau = 0.5 ###计算reward滑动平均的参数
        self.r_base = [0] ###r_base的列表，初始化为0
        # total learning step
        self.learn_step_counter = 0
        self.memory_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))
        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        self.saver = tf.train.Saver(max_to_keep=2)
        self.sess = tf.Session()
        # self.saver.restore(self.sess, restoreCheckpointFile)
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        tf.reset_default_graph()

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1,n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 200,256,\
                tf.random_normal_initializer(0., 0.03), tf.constant_initializer(0.01)  # config of layers
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            # # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            # with tf.variable_scope('l3'):
            #     w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
            #     b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            #
            # with tf.variable_scope('l3'):
            #     w3 = tf.get_variable('w3', [n_l2, n_l3], initializer=w_initializer, collections=c_names)
            #     b3 = tf.get_variable('b3', [1, n_l3], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, s_, done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_, [done]))
        # replace the old memory with new memory
        if self.memory_counter < self.memory_size-1:
            index = self.memory_counter
        else:
            index = np.random.randint(0, self.memory_size,1)
        # index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    # def store_transition(self, trans):
    #     if not hasattr(self, 'memory_counter'):
    #         self.memory_counter = 0
    #     for i in trans:
    #         transition = np.hstack((i[0], i[1], i[2], i[3], i[4]))
    #         # replace the old memory with new memory
    #         if self.memory_counter < self.memory_size - 1:
    #             index = self.memory_counter
    #         else:
    #             index = np.random.randint(0, self.memory_size, 1)
    #         # index = self.memory_counter % self.memory_size
    #         self.memory[index, :] = transition
    #         self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')
        # if self.learn_step_counter % 1000 == 0:
        #     self.saver.save(self.sess, saveCheckpointFile, global_step=self.learn_step_counter)
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features-1:-1],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        # c_r = batch_memory[:, self.n_features + 2]
        done = batch_memory[:, -1]
        ###计算reward值用滑动平均的方法


        # for i in range(len(reward)):
        #     if i == 0:
        #         self.r_base[0] = self.tau*self.r_base[0]+(1-self.tau)*reward[0]
        #     else:
        #         self.r_base.append(self.tau*self.r_base[-1]+(1-self.tau)*reward[i])

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) * done

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # cr_sum = np.sum(self.memory[:, self.n_features+2])##[self.n_features+2]
        # self.gama = self.gama + self.l_r * cr_sum/self.memory_size

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()




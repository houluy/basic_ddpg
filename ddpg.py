import tensorflow as tf
from itertools import chain
from collections import namedtuple
from collections.abc import MutableSequence
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import pathlib
import datetime

config = {
    "batch_size": 8,  # Size of a mini-batch
    "dropout": 0.01,
    "actor_lr": 0.01,  # lr = learning rate
    "critic_lr": 0.01,
    "actor_decay": 0.01,  # lr decay rate
    "critic_decay": 0.01,
    "ma_rate": 0.001,  # Moving average rate
    "regularizer_weight": 0.003,
    "huber_boundary": 100,  # Boundary of Huber loss
    "actor_layers": [4, 4],
    "critic_layers": [4, 4],
    "train_length": 50,  # Length of exp pool to start a train
    "gamma": 0.99,  # Gamma
}


FCLayer = namedtuple('FCLayer', (
    'name', 'layer', 'shape', 'regularizer', 'activation'
))


class GaussNoise:
    def __init__(self, action_dimension, episodes, mu=0, sigma=1):
        self.action_dimesion = action_dimension
        self.episodes = episodes
        self.mu = mu
        self.sigma = sigma

    def noise(self, batch_size, episode):
        return nr.normal(self.mu, self.sigma*(self.episodes - episode + 1e-8)/self.episodes, (batch_size, self.action_dimesion))


class DeepDeterministicPolicyGradient:
    def __init__(self, episodes):
        tf.reset_default_graph()
        self.state_shape = 1
        self.actor_opt_shape = self.action_shape = 2
        self.reward_shape = 1
        self.episodes = episodes
        self.critic_opt_shape = 1
        self.__dict__.update(config)
        self.actor_input = tf.placeholder(tf.float32, shape=(None, self.state_shape))
        self.critic_Q_input = tf.placeholder(tf.float32, shape=(None, 1))
        self.critic_state_input = tf.placeholder(
            tf.float32,
            shape=(
                None,
                self.state_shape
            )
        )
        self.critic_action_input = tf.placeholder(
            tf.float32,
            shape=(
                None,
                self.action_shape
            )
        )
        layers_dict = {}
        layer_type = ['actor', 'critic']
        layers_dict.update(zip(layer_type, [{}, {}]))
        layer_func = ['', 'target']
        for layer in layer_type:
            neuron_nums = getattr(self, layer + "_layers")
            for sublayer in layer_func:
                layers_dict[layer][sublayer] = [
                    FCLayer(name=layer + sublayer,
                            layer=i + 1,
                            shape=j,
                            regularizer=True,
                            activation=tf.nn.relu
                    ) for i, j in enumerate(neuron_nums)
                ]
                layers_dict[layer][sublayer].append(
                    FCLayer(name=layer + sublayer,
                            layer=-1,
                            shape=getattr(self, layer + "_opt_shape"),
                            regularizer=False,
                            activation=None
                            )
                )

        self.actor_layers = layers_dict['actor']['']
        self.actor_target_layers = layers_dict['actor']['target']
        self.critic_layers = layers_dict['critic']['']
        self.critic_target_layers = layers_dict['critic']['target']
        self.actor_model = self.build_model(
            self.actor_input,
            self.actor_layers,
            'actor'
        )
        self.critic_model = self.build_model(
            [self.critic_state_input, self.critic_action_input],
            self.critic_layers,
            'critic'
        )
        self.actor_target_model = self.build_model(
            self.actor_input,
            self.actor_target_layers,
            'actor',
        )
        self.critic_target_model = self.build_model(
            [self.critic_state_input, self.critic_action_input],
            self.critic_target_layers,
            'critic'
        )
        self.sess = tf.Session()
        self.critic_loss = self.square_loss(
            self.critic_Q_input,
            self.critic_model,
        )
        self.critic_optimizer = tf.train.AdamOptimizer(
            self.critic_lr
        ).minimize(self.critic_loss + sum(tf.get_collection('critic-losses')))

        self.critic_gradient = tf.gradients(
            self.critic_model, self.critic_action_input
        )
        self.critic_q = tf.reduce_mean(self.critic_model)
        # self.episode_pool = []  # Save experiences of each episodes

        self.exploration_noise = GaussNoise(self.action_shape, self.episodes)

        self.q_gradient_input = tf.placeholder(
            tf.float32, shape=[None, self.action_shape]
        )
        self.softmax_ipt = tf.placeholder(tf.float32, shape=(None, self.action_shape))
        self.softmax_op = tf.nn.softmax(self.softmax_ipt)
        self.actor_parameters = list(chain(*tf.get_collection('actor')))
        self.critic_parameters = list(chain(*tf.get_collection('critic')))
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.ma_rate)
        self.actor_target_update = self.ema.apply(self.actor_parameters)
        self.critic_target_update = self.ema.apply(self.critic_parameters)
        self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters, - self.q_gradient_input)
        self.actor_optimizer = tf.train.AdamOptimizer(
            self.actor_lr
        ).apply_gradients(
            zip(self.parameters_gradients, self.actor_parameters)
        )
        #self.reward = tf.Variable(0.)
        self.define_summaries()
        self.sess.run(tf.global_variables_initializer())
        self.critic_loss_record = []
        self.actor_j_record = []
        self._copy_weights('critic', 'target_critic')
        self._copy_weights('actor', 'target_actor')
        self.summary_path = pathlib.Path('summary')
        self.summary_writter = tf.summary.FileWriter(self.summary_path, self.sess.graph)
        self.summary_writter.flush()

    def define_summaries(self):
        episode_total_reward = tf.Variable(0.)
        loss = tf.Variable(0.)
        critic_output = tf.Variable(0.)
        episode_total_time_reward = tf.Variable(0.)
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.base_str = f'{time_str}-Summary'
        merged_summaries = [
            'Critic Output',
            # 'Average Q/Episode',
            # 'Maximum Q/Episode',
            # 'Minimum Q/Episode',
            # 'Total Q/Episode',
            'Average Loss',
        ]
        #loss_summary = f'{self.base_str}/Instant Loss/Training Round'
        merged_summaries = list(map(lambda x: f'{self.base_str}/{x}', merged_summaries))
        self.summaries = [
            critic_output,
            loss,
            episode_total_reward,
            episode_total_time_reward,
        ]
        self.placeholders = [tf.placeholder(tf.float32) for _ in merged_summaries]
        self.summary_assign_ops = [summary.assign(placeholder) for summary, placeholder in
                                   zip(self.summaries, self.placeholders)]
        for s_str, s in zip(merged_summaries, self.summaries):
            tf.summary.scalar(s_str, s)
        self.merged_summary = tf.summary.merge_all()

    @staticmethod
    def huber_loss(y_true, y_pred, clip_delta=1):
        error = y_true - y_pred
        cond = tf.keras.backend.abs(error) < clip_delta

        squared_loss = 0.5 * tf.keras.backend.square(error)
        linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
        return tf.reduce_mean(tf.where(cond, squared_loss, linear_loss))

    @staticmethod
    def square_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def _build_layer(self, ipt_layer, opt_layer):
        with tf.variable_scope(opt_layer.name, reuse=tf.AUTO_REUSE):
            ipt_layer = tf.layers.Flatten()(ipt_layer)
            ipt_size = ipt_layer.get_shape()[-1]
            weight_shape = [ipt_size, opt_layer.shape]
            weights, biases = self.gen_weights(
                opt_layer.name,
                opt_layer.name + str(opt_layer.layer),
                weight_shape,
                bias_shape=[opt_layer.shape],
                regularizer=opt_layer.regularizer,
                wl=self.regularizer_weight,
            )
            tf.add_to_collections(opt_layer.name, [weights, biases])
            clayer = tf.add(tf.matmul(ipt_layer, weights), biases)
            if opt_layer.activation is not None:
                clayer = opt_layer.activation(clayer)
                clayer = tf.nn.dropout(clayer, 1 - self.dropout)
        return clayer

    @staticmethod
    def gen_weights(model_name, scope_name, shape, bias_shape, stddev=.3, bias=.3,
                    regularizer=None, wl=None):
        weight_init = tf.truncated_normal_initializer(dtype=tf.float32,
                                                      stddev=stddev)
        bias_init = tf.constant_initializer(bias)
        weights = tf.get_variable('{}-weights'.format(scope_name),
                                  shape=shape, initializer=weight_init)
        biases = tf.get_variable('{}-biases'.format(scope_name),
                                 shape=bias_shape, initializer=bias_init)
        if regularizer is not None:
            weights_loss = tf.multiply(tf.nn.l2_loss(weights), wl,
                                       name='weights-loss')
            tf.add_to_collection('{}-losses'.format(model_name), weights_loss)
        return weights, biases

    def build_model(self, ipt, layers, typ='actor'):
        if isinstance(ipt, MutableSequence):
            current = tf.concat(ipt, axis=1)
        else:
            current = ipt
        for layer in layers:
            current = self._build_layer(current, layer)
        if typ == 'actor':
            current = tf.nn.softmax(current)
        return current

    def _copy_weights(self, src_name, dest_name):
        m1 = [t for t in tf.trainable_variables() if t.name.startswith(src_name)]
        m1 = sorted(m1, key=lambda v: v.name)
        m2 = [t for t in tf.trainable_variables() if t.name.startswith(dest_name)]
        m2 = sorted(m2, key=lambda v: v.name)

        ops = []
        for t1, t2 in zip(m1, m2):
            ops.append(t2.assign(t1))
        self.sess.run(ops)

    def __del__(self):
        try:
            self.sess.close()
        except AttributeError:
            print('Something wrong before the session was created')

    def train_networks(self, minibatch, epoch):
        # self.timer()
        batch_state, batch_action, batch_reward, batch_nstate, batch_done = minibatch
        # self.timer()
        batch_next_action = self.sess.run(self.actor_target_model, feed_dict={
            self.actor_input: batch_nstate,
        })
        # self.timer()
        batch_target_q = self.sess.run(self.critic_target_model, feed_dict={
            self.critic_state_input: batch_nstate,
            self.critic_action_input: batch_next_action,
        })
        # self.timer()
        y = []
        for ind in range(self.batch_size):
            if batch_done[ind]:
                y.append(batch_reward[ind])
            else:
                y.append(batch_reward[ind] + self.gamma * batch_target_q[ind])
        # self.timer()
        y = np.resize(y, [self.batch_size, 1])
        _, critic_loss, critic_q = self.sess.run(
            [self.critic_optimizer, self.critic_loss, self.critic_q],
            feed_dict={
                self.critic_state_input: batch_state,
                self.critic_action_input: batch_action,
                self.critic_Q_input: y
            }
        )
        # self.timer()
        # Update gradient
        batch_action = self.sess.run(self.actor_model, feed_dict={
            self.actor_input: batch_state,
        })
        # self.timer()
        q_gradients_batch = self.sess.run(self.critic_gradient, feed_dict={
            self.critic_state_input: batch_state,
            self.critic_action_input: batch_action,
        })[0]
        # self.timer()
        # Update actor model
        self.sess.run(self.actor_optimizer, feed_dict={
            self.q_gradient_input: q_gradients_batch,
            self.actor_input: batch_state,
        })
        # self.timer()
        # Update target model
        self.sess.run([self.actor_target_update, self.critic_target_update])
        # self.critic_loss_record.append(critic_loss)
        # self.timer()
        summaries = [
            critic_q,
            critic_loss,
        ]
        for ind, i in enumerate(summaries):
            self.sess.run(self.summary_assign_ops[ind], feed_dict={
                self.placeholders[ind]: i
            })
        summary = self.sess.run(self.merged_summary)
        self.summary_writter.add_summary(summary, epoch)
        # self.timer()
        # self.timer.reset()
        return critic_loss

    def __call__(self, state, batch_size, train=False, episode=0):
        if train:
            noise = self.exploration_noise.noise(batch_size, episode)
        else:
            noise = np.zeros((batch_size, self.action_shape))

        action = self.sess.run(self.actor_model, feed_dict={
            self.actor_input: state,
        })
        return self.sess.run(self.softmax_op, feed_dict={
            self.softmax_ipt: action + noise,
        })

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()

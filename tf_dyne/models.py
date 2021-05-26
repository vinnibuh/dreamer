import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd

import tools


class DenseActionEncoder(tools.Module):

    def __init__(self, traj_size, embed_size,
                 act=tf.nn.selu):
        self._traj_size = traj_size
        self._embed_size = embed_size
        self.hidden_size = 400
        self._act = act

    def __call__(self, actions):
        x = actions
        x = self.get(f'h1', tfkl.Dense, self._hidden_size, self._act)(x)
        mu = self.get(f'hmu', tfkl.Dense, self._embed_size)(x)
        std = self.get(f'hstd', tfkl.Dense, self._embed_size)(x)
        return mu, std


class ConvStateEncoder(tools.Module):

    def __init__(self, stack, hidden_size,
                 act=tf.nn.selu, depth=32, shape=(64, 64, 3)):
        self._act = act
        self._stack = stack
        self._depth = 32
        self._hidden_size = hidden_size
        self._shape = shape

    def __call__(self, obs):
        kwargs = dict(strides=2, activation=self._act)
        x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))

        x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
        x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
        x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
        x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
        shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
        return tf.reshape(x, shape)


class DenseDecoder(tools.Module):

    def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
        self._shape = shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
        x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
        if self._dist == 'normal':
            return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
        if self._dist == 'binary':
            return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
        raise NotImplementedError(self._dist)


class ActionDecoder(tools.Module):

    def __init__(
            self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
            min_std=1e-4, init_std=5, mean_scale=5):
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, features):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        if self._dist == 'tanh_normal':
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'onehot':
            x = self.get(f'hout', tfkl.Dense, self._size)(x)
            dist = tools.OneHotDist(x)
        else:
            raise NotImplementedError(dist)
        return dist

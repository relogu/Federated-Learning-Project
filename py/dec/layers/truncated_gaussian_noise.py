import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend
from tensorflow.keras.backend import random_normal


class TruncatedGaussianNoise(Layer):
    """"""
    
    def __init__(self, stddev, rate: float = 0.5, seed=None, **kwargs):
        super(TruncatedGaussianNoise, self).__init__(**kwargs)
        self.stddev = stddev
        self.rate = rate
        self.seed = seed
    
    def call(self, inputs, training=None):
        
        def noised():
            noised = inputs + random_normal(
                shape=tf.shape(inputs),
                mean=0.,
                stddev=self.stddev,
                dtype=inputs.dtype
            )
            zeros = tf.zeros(shape=tf.shape(inputs))
            ones = tf.ones(shape=tf.shape(inputs))
            condition = tf.random.uniform(shape=tf.shape(inputs)) <= self.rate
            noised = tf.where(condition, inputs, noised)
            neg_condition = noised < 0.0
            greater_than_one_condition = noised > 1.0
            noised = tf.where(neg_condition, zeros, noised)
            noised = tf.where(greater_than_one_condition, ones, noised)
            return noised
        
        return backend.in_train_phase(noised, inputs, training=training)
    
    def get_config(self):
        config = {'stddev': self.stddev, 'seed': self.seed}
        base_config = super(TruncatedGaussianNoise, self).get_config()
        return dict(list(base_config.items())) + list(config.items())
    
    def compute_output_shape(self, input_shape):
        return input_shape

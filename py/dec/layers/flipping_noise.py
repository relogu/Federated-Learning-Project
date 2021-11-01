#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer

class FlippingNoise(Layer):
    def __init__(self, up_frequencies=None, rate=0.01, *args, **kwargs):
        super(FlippingNoise, self).__init__(*args, **kwargs)
        self.up_frequencies = up_frequencies
        self.rate = rate

    def build(self, input_shape):
        assert len(input_shape) == 2
        if self.up_frequencies is None:
            self.up_frequencies = np.array([0.5]*input_shape[1])
        else:
            assert input_shape[1] == len(self.up_frequencies)
        self.down_frequencies = 1-self.up_frequencies
        self.probs = tf.transpose(tf.stack((self.down_frequencies, self.up_frequencies)))
        
    def call(self,
             inputs, 
             training=False # Only add noise in training!
             ):
        if training:
            samples = tf.cast(tf.transpose(tf.random.categorical(tf.math.log(self.probs), tf.shape(inputs)[0])), tf.float32)
            return tf.where(tf.random.uniform(shape=tf.shape(inputs))>self.rate, inputs, samples)
        else:
            return inputs # return inputs during inference

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Aug 4 10:37:10 2021

@author: relogu
"""

from typing import List
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer

class FlippingNoise(Layer):
    def __init__(self,
                 b_idx: List[int] = None,
                 up_frequencies: List[float] = None,
                 rate: float = 0.01,
                 *args, **kwargs):
        super(FlippingNoise, self).__init__(*args, **kwargs)
        self.up_frequencies = up_frequencies
        self.rate = rate
        self.b_idx = b_idx

    def build(self, input_shape):
        assert len(input_shape) == 2
        if self.b_idx is not None:
            shape_of_change = len(self.b_idx)
            assert shape_of_change <= input_shape[1]
            setA = set(list(range(input_shape[1])))
            setB = set(self.b_idx)
            onlyInA = setA.difference(setB)
            self.nb_idx = [int(i) for i in list(onlyInA)]
            print('Non-binary indices: {}'.format(self.nb_idx))
        else:
            shape_of_change = input_shape[1]
            self.b_idx = list(range(shape_of_change))
            self.nb_idx = None
        if self.up_frequencies is None:
            self.up_frequencies = np.array([0.5]*shape_of_change)
        else:
            assert shape_of_change == len(self.up_frequencies)
        self.down_frequencies = 1-self.up_frequencies
        self.probs = tf.transpose(tf.stack((self.down_frequencies, self.up_frequencies)))
        
    def call(self,
             inputs, 
             training=False # Only add noise in training!
             ):
        if training:
            to_change = tf.gather(inputs, indices=self.b_idx, axis=1)
            samples = tf.cast(tf.transpose(tf.random.categorical(tf.math.log(self.probs), tf.shape(to_change)[0])), tf.float32)
            condition = tf.random.uniform(shape=tf.shape(to_change)) >= self.rate
            b_changed = tf.where(condition, to_change, samples)
            indices = tf.expand_dims(self.b_idx, axis=1)
            binary = tf.transpose(tf.scatter_nd(indices, tf.transpose(b_changed), tf.shape(tf.transpose(inputs))))
            if self.nb_idx is None:
                output = binary
            else:
                not_to_change = tf.gather(inputs, indices=self.nb_idx, axis=1)
                indices = tf.expand_dims(self.nb_idx, axis=1)
                n_binary = tf.transpose(tf.scatter_nd(indices, tf.transpose(not_to_change), tf.shape(tf.transpose(inputs))))
                output = tf.add(binary, n_binary)
            return output
        else:
            return inputs # return inputs during inference

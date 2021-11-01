import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    # r_inputs = K.round(inputs) # can't be done in loss fn
    targets = K.flatten(targets)
    c_targets = tf.cast(targets, tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(c_targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))
    dice = 1 - (2*intersection + smooth) / (K.sum(c_targets) + K.sum(inputs) + smooth)
    bce = binary_crossentropy(targets, inputs)
    dice_bce = bce + dice
    
    return dice_bce
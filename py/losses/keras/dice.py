import tensorflow as tf
import tensorflow.keras.backend as K

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    # inputs = K.round(inputs) # can't be done in loss fn
    targets = K.flatten(targets)
    targets = tf.cast(targets, tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice
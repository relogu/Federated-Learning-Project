import tensorflow as tf
import tensorflow.keras.backend as K

def IoULoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    c_targets = tf.cast(targets, tf.float32)
    
    intersection = K.sum(K.dot(tf.expand_dims(c_targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))
    total = K.sum(c_targets) + K.sum(inputs)
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou
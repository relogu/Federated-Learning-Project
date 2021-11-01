import tensorflow as tf
import tensorflow.keras.backend as K

ALPHA = 0.8
GAMMA = 2
# very good
def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    targets = tf.cast(targets, tf.float32)
    
    bce = K.binary_crossentropy(targets, inputs)
    bce_exp = K.exp(-bce)
    focal_loss = K.mean(alpha * K.pow((1-bce_exp), gamma) * bce)
    
    return focal_loss
import tensorflow as tf
import tensorflow.keras.backend as K


def IoUCosineLoss(targets, inputs, smooth=1e-6):
    """
    Cosine Distance is a classic vector distance metric that 
    is used commonly when comparing Bag of Words representations 
    in NLP problems. The distance is calculated by finding the 
    cosine angle between the two vectors
    """
    
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    c_targets = tf.cast(targets, tf.float32)
    
    y_true = tf.linalg.l2_normalize(c_targets, axis=0)
    y_pred = tf.linalg.l2_normalize(inputs, axis=0)

    intersection = K.sum(K.dot(tf.expand_dims(c_targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))
    total = K.sum(c_targets) + K.sum(inputs)
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou - tf.reduce_sum(y_true * y_pred)

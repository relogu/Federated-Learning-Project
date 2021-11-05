import tensorflow as tf
import tensorflow.keras.backend as K


def CosineSimilarityLoss(targets, inputs, smooth=1e-6):
    """
    Cosine Distance is a classic vector distance metric that 
    is used commonly when comparing Bag of Words representations 
    in NLP problems. The distance is calculated by finding the 
    cosine angle between the two vectors
    """
    
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    targets = tf.cast(targets, tf.float32)
    
    y_true = tf.linalg.l2_normalize(targets, axis=0)
    y_pred = tf.linalg.l2_normalize(inputs, axis=0)
    return -tf.reduce_sum(y_true * y_pred)
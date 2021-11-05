import tensorflow as tf
import tensorflow.keras.backend as K

ALPHA = 0.8
GAMMA = 2
# very good


def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    """
    Focal Loss was introduced by Lin et al of Facebook AI Research
    in 2017 as a means of combatting extremely imbalanced datasets
    where positive cases were relatively rare. Their paper "Focal Loss
    for Dense Object Detection" is retrievable here: 
    https://arxiv.org/abs/1708.02002. 
    In practice, the researchers used an alpha-modified version of
    the function so I have included it in this implementation.
    """

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    targets = tf.cast(targets, tf.float32)

    bce = K.binary_crossentropy(targets, inputs)
    bce_exp = K.exp(-bce)
    focal_loss = K.mean(alpha * K.pow((1-bce_exp), gamma) * bce)

    return focal_loss

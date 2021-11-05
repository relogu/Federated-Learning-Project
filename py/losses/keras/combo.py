import tensorflow as tf
import tensorflow.keras.backend as K

ALPHA = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5  # weighted contribution of modified CE loss compared to Dice loss


def ComboLoss(targets, inputs, eps=1e-9, smooth=1e-6):
    """
    This loss was introduced by Taghanaki et al in their paper
    "Combo loss: Handling input and output imbalance in multi-organ segmentation",
    retrievable here: https://arxiv.org/abs/1805.02798.
    Combo loss is a combination of Dice Loss and a modified Cross-Entropy function
    that, like Tversky loss, has additional constants which penalise either false
    positives or false negatives more respectively.
    """
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    c_targets = tf.cast(targets, tf.float32)

    intersection = K.sum(K.dot(tf.expand_dims(c_targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))
    dice = (2*intersection + smooth) / \
        (K.sum(c_targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (ALPHA * ((c_targets * K.log(inputs)) +
             ((1 - ALPHA) * (1.0 - c_targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

    return combo

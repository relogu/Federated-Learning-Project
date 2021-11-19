import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy


def DiceBCELoss(targets, inputs, smooth=1e-6):
    """
    This loss combines Dice loss with the standard binary cross-entropy (BCE)
    loss that is generally the default for segmentation models.
    Combining the two methods allows for some diversity in the loss,
    while benefitting from the stability of BCE.
    Dice-Coefficient for probability is computed as:
    DC = |A*B|/(|A|+|B|-|A*B|), where |A*B| is intersection
    Dice-Coefficient loss is 1-DC, that could be rewritten as:
    DCl = (|A|+|B|)/(|A|+|B|-|A*B|)
    """

    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    # r_inputs = K.round(inputs) # can't be done in loss fn
    targets = K.flatten(targets)
    c_targets = tf.cast(targets, tf.float32)
    intersection = K.sum(K.dot(tf.expand_dims(c_targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))

    # OLD
    # dice = 1 - (2*intersection + smooth) / \
    #     (K.sum(c_targets) + K.sum(inputs) + smooth)
    dice = (K.sum(c_targets) + K.sum(inputs) + smooth)/ \
        (K.sum(c_targets) + K.sum(inputs) - intersection + smooth)
    bce = binary_crossentropy(targets, inputs)
    dice_bce = bce + dice

    return dice_bce

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error


def DiceMSELoss(targets, inputs, smooth=1e-6):
    """
    This loss combines Dice loss with the standard mean squared error (MSE)
    loss that is generally the default for segmentation models.
    Combining the two methods allows for some diversity in the loss,
    while benefitting from the stability of MSe. 
    """

    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    # r_inputs = K.round(inputs) # can't be done in loss fn
    targets = K.flatten(targets)
    c_targets = tf.cast(targets, tf.float32)

    intersection = K.sum(K.dot(tf.expand_dims(c_targets, axis=0),
                               K.transpose(tf.expand_dims(inputs, axis=0))))
    dice = 1 - (2*intersection + smooth) / \
        (K.sum(c_targets) + K.sum(inputs) + smooth)
    mse = mean_squared_error(targets, inputs)
    dice_mse = mse + dice

    return dice_mse
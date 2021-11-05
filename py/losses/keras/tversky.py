import tensorflow as tf
import tensorflow.keras.backend as K

ALPHA = 0.5
BETA = 0.5


def TverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    """
    This loss was introduced in "Tversky loss function for image segmentation
    using 3D fully convolutional deep networks", retrievable here: 
    https://arxiv.org/abs/1706.05721.
    It was designed to optimise segmentation on imbalanced medical datasets
    by utilising constants that can adjust how harshly different types of
    error are penalised in the loss function. From the paper:
    
    ... in the case of α=β=0.5 the Tversky index simplifies to be the same
    as the Dice coefficient, which is also equal to the F1 score.
    With α=β=1, Equation 2 produces Tanimoto coefficient, and setting α+β=1
    produces the set of Fβ scores. Larger βs weigh recall higher than precision
    (by placing more emphasis on false negatives).
    
    To summarise, this loss function is weighted by the constants 'alpha' and
    'beta' that penalise false positives and false negatives respectively to
    a higher degree in the loss function as their value is increased. The beta
    constant in particular has applications in situations where models can
    obtain misleadingly positive performance via highly conservative prediction.
    You may want to experiment with different values to find the optimum.
    With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
    """

    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    targets = tf.cast(targets, tf.float32)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))

    tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return 1 - tversky

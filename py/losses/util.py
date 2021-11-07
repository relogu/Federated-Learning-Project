from tensorflow.python.keras.losses import mean_squared_error, binary_crossentropy
from .keras import (DiceMSELoss, ComboLoss, CosineSimilarityLoss,
                    DiceBCELoss, DiceLoss, FocalLoss, FocalTverskyLoss,
                    IoUCosineLoss, IoUDiceLoss, IoULoss, TverskyLoss)
KERAS_LOSSES_DICT = {
    'dice_mse': DiceMSELoss,
    'combo': ComboLoss,
    'cosine_smilarity': CosineSimilarityLoss,
    'dice_bce': DiceBCELoss,
    'dice': DiceLoss,
    'focal': FocalLoss,
    'focal_tversky': FocalTverskyLoss,
    'iou_cosine': IoUCosineLoss,
    'iou_dice': IoUDiceLoss,
    'iou': IoULoss,
    'tversky': TverskyLoss,
    'mse': mean_squared_error,
    'bce': binary_crossentropy,
}

def get_keras_loss_names():
    return KERAS_LOSSES_DICT.keys()

def get_keras_loss(name: str):
    return KERAS_LOSSES_DICT[name]

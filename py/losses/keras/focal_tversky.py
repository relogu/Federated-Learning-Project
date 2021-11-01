import tensorflow as tf
import tensorflow.keras.backend as K

ALPHA = 0.5
BETA = 0.5
GAMMA = 1

def FocalTverskyLoss(targets, inputs, alpha=ALPHA, beta=BETA, gamma=GAMMA, smooth=1e-6):
    
        #flatten label and prediction tensors
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        targets = tf.cast(targets, tf.float32)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        focal_tversky = K.pow((1 - tversky), gamma)
        
        return focal_tversky
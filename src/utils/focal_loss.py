import tensorflow as tf
from tensorflow.keras import backend as K

def sparse_categorical_focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for multi-class classification with integer labels.
    alpha: balancing factor (helps with class imbalance)
    gamma: focusing parameter (helps with hard-to-classify examples)
    """
    def focal_loss_fixed(y_true, y_pred):
        # 1. Convert integer labels to one-hot encoding
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        
        # 2. Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # 3. Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # 4. Calculate Focal Loss component
        # (1 - p)^gamma helps the model focus on samples it is NOT confident about
        loss = alpha * K.pow(1.0 - y_pred, gamma) * cross_entropy
        
        # 5. Sum across classes and average across batch
        return K.mean(K.sum(loss, axis=-1))
    
    return focal_loss_fixed
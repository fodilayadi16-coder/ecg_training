import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, Activation, 
                                     Add, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from utils.focal_loss import sparse_categorical_focal_loss

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First Convolution
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second Convolution
    x = Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection (adjust dimensions if stride > 1 or filter size changes)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_beat_resnet(input_shape=(360, 1), num_classes=5):
    inputs = Input(shape=input_shape)
    
    # --- Initial Stem ---
    # To improve S-class recall, consider changing strides to 1 
    # if this version still underperforms.
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)
    
    # --- ResNet Blocks ---
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # --- Final Classifier ---
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=3e-4), 
        # Using the Focal Loss parameters we discussed for S-class boost
        loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model  
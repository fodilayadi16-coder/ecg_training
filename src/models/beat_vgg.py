from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Activation, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from utils.focal_loss import sparse_categorical_focal_loss

def build_beat_vgg(input_shape=(360, 1), num_classes=5):
    inputs = Input(shape=input_shape)

# Block 1: Capture fine morphological details
    x = Conv1D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x) # Resolution: 180

    # Block 2: Start grouping features
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x) # Resolution: 90

    # Block 3: Higher-level patterns (QRS complex relationships)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x) # Resolution: 45

    # Classifier Head
    x = GlobalAveragePooling1D()(x) # Prevents overfitting better than Flatten
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4), # VGG usually requires a lower LR than ResNet
        loss=sparse_categorical_focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    
    return model

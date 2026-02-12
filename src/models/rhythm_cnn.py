from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D

def build_rhythm_cnn(input_shape=(1800, 1)):
    model = Sequential([
        Conv1D(32, 15, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(4),

        Conv1D(64, 11, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(4),

        Conv1D(128, 7, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # AF vs Normal
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

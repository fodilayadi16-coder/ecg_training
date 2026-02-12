import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.model_selection import train_test_split
from models.beat_cnn import build_beat_cnn
from training.callbacks import make_callbacks
# from utils.balancing import get_class_weights
from utils.oversampling import moderate_ros

X = np.load("data/processed/beat_X.npy") 
y = np.load("data/processed/beat_y.npy") 

# Shape of original processed data
print(X.shape, y.shape)
print(np.unique(y, return_counts=True))

X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8) # Normalizing the data to have mean 0 and std 1 for each sample, adding a small epsilon to avoid division by zero

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

# Apply moderate ROS only on training set to balance the dataset
X_train_resampled, y_train_resampled = moderate_ros(X_train, y_train)

print(np.unique(y_train_resampled, return_counts=True))

model = build_beat_cnn()
model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=128,
    # class_weight = get_class_weights(y_train), we don't use it since we are using ROS
    callbacks=make_callbacks("beat")
)

model.save("models/beat_final.h5")














import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.model_selection import train_test_split
from models.rhythm_cnn import build_rhythm_cnn
from training.callbacks import make_callbacks
from utils.balancing import get_class_weights

# ── Load pre-processed data (already filtered by clean_ecg) ──
X = np.load("data/processed/rhythm_X.npy")
y = np.load("data/processed/rhythm_y.npy")

print("Original classes:", np.unique(y, return_counts=True))
print("Input shape:", X.shape)

# Split by index (stratified)
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# With 1M+ windows, class weights alone handle the 2.7:1 imbalance — no need for ROS
class_weights = get_class_weights(y_train)
class_weights[1] = class_weights[1] * 1.5  # boost AF weight
print("Class weights:", class_weights)

# ── Build residual CNN ───────────────────────
input_shape = X_train.shape[1:]  # e.g. (1800, 1)
model = build_rhythm_cnn(input_shape=input_shape, num_classes=2, dropout_rate=0.3)
model.summary()

# ── Train ────────────────────────────────────
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=128,
    class_weight=class_weights,
    callbacks=make_callbacks("rhythm"),
)

model.save("models/rhythm_final.h5")
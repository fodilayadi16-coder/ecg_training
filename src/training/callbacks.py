import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def make_callbacks(name):
    save_path = f"models/{name}_best.h5"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]

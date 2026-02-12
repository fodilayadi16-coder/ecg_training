from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def make_callbacks(name):
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(f"models/{name}_best.h5", monitor="val_loss", save_best_only=True, verbose=1)
    ]

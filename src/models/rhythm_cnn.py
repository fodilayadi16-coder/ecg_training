import os
import tensorflow as tf

num_threads = os.cpu_count()

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.optimizer.set_jit(True)  # Optional speed boost

print(f"Using {num_threads} threads")

from tensorflow.keras import layers, models, regularizers


def _residual_block(x, filters, kernel_size, dilation_rate=1, dropout_rate=0.3):
    """Residual block with dilated convolution."""
    shortcut = x

    out = layers.Conv1D(
        filters, kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    out = layers.SpatialDropout1D(dropout_rate)(out)

    out = layers.Conv1D(
        filters, kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(1e-4),
    )(out)
    out = layers.BatchNormalization()(out)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    out = layers.Add()([shortcut, out])
    out = layers.ReLU()(out)
    return out


def build_rhythm_cnn(input_shape, num_classes=2, dropout_rate=0.3):
    """
    Multi-scale dilated residual CNN for rhythm classification.

    Small kernels (3) with increasing dilation rates replace large kernels,
    giving the same receptive field while being more parameter-efficient
    and less prone to overfitting.

    Receptive field comparison:
        - Old: kernel=15 → 15 samples
        - New: kernel=3, dilation=1,2,4,8 → effective RF = 31 samples
    """
    inp = layers.Input(shape=input_shape)

    # ── Initial feature extraction (small kernel) ────────
    x = layers.Conv1D(32, 3, padding="same",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ── Dilated residual blocks (exponentially growing receptive field) ──
    x = _residual_block(x, filters=32, kernel_size=3, dilation_rate=1,
                        dropout_rate=dropout_rate)
    x = layers.MaxPooling1D(2)(x)

    x = _residual_block(x, filters=64, kernel_size=3, dilation_rate=2,
                        dropout_rate=dropout_rate)
    x = layers.MaxPooling1D(2)(x)

    x = _residual_block(x, filters=128, kernel_size=3, dilation_rate=4,
                        dropout_rate=dropout_rate)
    x = layers.MaxPooling1D(2)(x)

    x = _residual_block(x, filters=128, kernel_size=3, dilation_rate=8,
                        dropout_rate=dropout_rate)

    # ── Aggregation ──────────────────────────────────────
    # Global pooling: both avg and max for richer representation
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # ── Classifier head ──────────────────────────────────
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    if num_classes == 2:
        out = layers.Dense(1, activation="sigmoid")(x)
    else:
        out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out, name="rhythm_dilated_res_cnn")

    loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=["accuracy"],
    )

    return model

import tensorflow as tf
import os

# ===== Paths =====
h5_model_path = "models/cnn/cnn_beat_best.h5"
tflite_output_path = "export/ecg_model.tflite"

# ===== Load Keras Model (do not compile) =====
model = tf.keras.models.load_model(h5_model_path, compile=False)

# Build a concrete function with training=False to avoid ReadVariableOp issues
# that can occur when converting models with BatchNormalization/variables.
input_shape = model.inputs[0].shape
input_dtype = model.inputs[0].dtype
input_spec = tf.TensorSpec(shape=input_shape, dtype=input_dtype)

@tf.function
def serving_fn(x):
    return model(x, training=False)

concrete_func = serving_fn.get_concrete_function(input_spec)

# ===== Convert to TFLite from the concrete function =====
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite_model = converter.convert()
except Exception as e:
    print("Conversion failed:", e)
    raise

# ===== Save TFLite Model =====
os.makedirs(os.path.dirname(tflite_output_path), exist_ok=True)
with open(tflite_output_path, "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete! Saved to:", tflite_output_path)
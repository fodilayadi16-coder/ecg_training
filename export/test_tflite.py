import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="export/ecg_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])

# Create dummy input (CHANGE shape to match your model!)
dummy_input = np.random.rand(1, 360, 1).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])

print("Output:", output) # output (Input shape: [  1 360   1] Output: [[0.6260501  0.07631013 0.19136116 0.02864143 0.07763717]])
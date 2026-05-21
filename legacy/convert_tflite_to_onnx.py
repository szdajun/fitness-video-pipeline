import tensorflow as tf
import tf2onnx
import os
import numpy as np

tflite_path = 'F:/wkspace/fitness-video-pipeline/models/pose_landmark_full.tflite'
onnx_path = 'F:/wkspace/fitness-video-pipeline/models/pose_landmark_full.onnx'

print('Loading TFLite model...')
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()

input_details = interp.get_input_details()
output_details = interp.get_output_details()

print(f'Input tensors ({len(input_details)}):')
for inp in input_details:
    print(f'  name={inp["name"]}, shape={inp["shape"]}, dtype={inp["dtype"]}')
print(f'Output tensors ({len(output_details)}):')
for out in output_details:
    print(f'  name={out["name"]}, shape={out["shape"]}, dtype={out["dtype"]}')

# Create a dummy input matching the expected shape
dummy_input = np.zeros(input_details[0]['shape'], dtype=np.float32)

# Convert using tf2onnx
print('Converting to ONNX...')

# Use from_tflite with model content
with open(tflite_path, 'rb') as f:
    tflite_content = f.read()

input_names = [inp['name'] for inp in input_details]
output_names = [out['name'] for out in output_details]

model_proto, _ = tf2onnx.convert.from_tflite(
    tflite_content,
    input_names=input_names,
    output_names=output_names,
    output_path=onnx_path
)

print(f'ONNX model saved: {os.path.getsize(onnx_path)} bytes')
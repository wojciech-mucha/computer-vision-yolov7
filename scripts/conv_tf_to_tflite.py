import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('weights_custom/')
tflite_model = converter.convert()

with open('yolov7-tiny_custom_pack10.tflite', 'wb') as f:
  f.write(tflite_model)

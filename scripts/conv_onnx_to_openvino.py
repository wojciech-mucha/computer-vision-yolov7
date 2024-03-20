import openvino as ov

model = ov.convert_model('yolov7-tiny_pack10_v2_exp4.onnx')
# serialize model for saving IR
ov.save_model(model, 'model/yolov7-tiny_pack10_v2_exp4.xml')
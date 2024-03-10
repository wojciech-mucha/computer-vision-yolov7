# https://github.com/VikasOjha666/yolov7_to_tflite/blob/main/yoloV7_to_TFlite%20.ipynb

import argparse
import time
from pathlib import Path
import cv2
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.torch_utils import time_synchronized

i_ = 0

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="yolov7-tiny_custom_pack10.tflite")


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny_custom_pack10.tflite', help='model.tflite path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()

    t0 = time.time()
    # Name of the classes according to class indices.
    names = ['pack10']

    # Creating random colors for bounding box visualization.
    colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}
    pathlist = Path(opt.source).glob('**/*.jpg')
    for path in pathlist:
        i_ = i_ + 1
        print(f"{i_}: {path}")
        # Load and preprocess the image.
        img = cv2.imread(filename=str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = img.copy()
        image, ratio, dwdh = letterbox(im=image, auto=False)
        image = image.transpose((2, 0, 1))

        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        # Allocate tensors.
        t1 = time_synchronized()
        interpreter.allocate_tensors()
        # Get input and output tensors.
        t2 = time_synchronized()
        input_details = interpreter.get_input_details()
        t3 = time_synchronized()
        output_details = interpreter.get_output_details()
        t4 = time_synchronized()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], im)
        t5 = time_synchronized()
        interpreter.invoke()
        t6 = time_synchronized()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        t7 = time_synchronized()

        ori_images = [img.copy()]

        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(output_data):
            image = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = names[cls_id]
            color = colors[name]
            name += ' ' + str(score)
            cv2.rectangle(image, box[:2], box[2:], color, 2)
            cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)  

        prediction = Image.fromarray(ori_images[0])
        # prediction.save(f'./runs/detect_tflite/{i}.jpg')
        cv2.imwrite(filename=f'./runs/detect_tflite/{i_}.jpg', img=ori_images[0])
        # prediction.show()
        print(f'Done. ({(1E3 * (t7 - t1)):.1f} ms) Inference, ({(1E3 * (t2 - t1)):.1f}ms) allocate_tensors; ({(1E3 * (t3 - t2)):.1f}ms) get_input_details; ({(1E3 * (t4 - t3)):.1f}ms) get_output_details; ({(1E3 * (t5 - t4)):.1f}ms) set_tensor; ({(1E3 * (t6 - t5)):.1f}ms) invoke; ({(1E3 * (t7 - t6)):.1f}ms) get_tensor')

    print(f'Done. ({time.time() - t0:.3f}s)')

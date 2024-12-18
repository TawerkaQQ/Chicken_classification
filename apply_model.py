import time

import cv2
import onnx
import numpy as np
import torch
import torchvision

from onnxruntime import InferenceSession
from collections import Counter




def main(chicken_switch: int = 20, frame_for_clf: int = 5) -> None:

    preds_list = []
    frame_count = 0

    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(2)

    if not (cam1.isOpened() or cam2.isOpened()):
        print("Can't open camera")
        exit()

    onnx_model = onnx.load('./clf_model.onnx')
    onnx.checker.check_model(onnx_model)

    session = InferenceSession('./clf_model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    while True:

        ret, frame = cam1.read()
        ret2, frame2 = cam2.read()

        frame = cv2.resize(frame,(224, 224))
        frame2 = cv2.resize(frame2, (224, 224))

        if not (ret or ret2):
            print("Can't read frame")
            break

        cv2.imshow('cam1', frame)
        cv2.imshow('cam2', frame2)

        frame_count += 1

        if frame_count % frame_for_clf == 0:

            #res for camera 1
            result = session.run([output_name], {input_name: frame.transpose(2, 0, 1)[np.newaxis, ...].astype('float32')})
            prediction_cam1 = int(np.argmax(np.array(result).squeeze(), axis=0))

            # res for camera 2
            result2 = session.run([output_name], {input_name: frame2.transpose(2, 0, 1)[np.newaxis, ...].astype('float32')})
            prediction_cam2 = int(np.argmax(np.array(result2).squeeze(), axis=0))

            preds_list.append(prediction_cam1)
            preds_list.append(prediction_cam2)

            frame_count = 0

            print(preds_list)

        if len(preds_list) == 10:

            pred_class = Counter(preds_list).most_common(1)[0][0]

            print('max of preds list:', pred_class)

            preds_list.clear()
            time.sleep(chicken_switch)

        if cv2.waitKey(1) == ord('q'):
            break

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

    return None


if __name__ == '__main__':

    main(chicken_switch = 20, frame_for_clf= 5)
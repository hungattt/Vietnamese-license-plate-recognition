import cv2 as cv
from imutils import perspective
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import time
def detec(frame):
    bd =time.time()
    net = cv.dnn_DetectionModel('./setup/yolov4-custom.cfg',
                                './setup/yolov4-custom_best_dot1.weights')
    net.setInputSize(416, 416)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    kt=time.time()
    print("thoi gian detec bien so : ",kt-bd )
    print(boxes)
    print(len(boxes))
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        print(boxes[i])
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        print(poly)
        poly = np.array(poly).astype(int)
        LpRegion = perspective.four_point_transform(frame, poly)
        cv.imwrite(f'./Anh/crop{i}.jpg', LpRegion)
        frame = cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
        frame = cv.circle(frame, (x + w, y + h), 3, (0, 0, 255), -1)
        cv.rectangle(frame,boxes[i], color=(0, 255, 0), thickness=2)
        print(200 * '-')
        # --------------------------------doc bien so--------------------------------------
        bd1=time.time()
        ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
        img_path = f'./Anh/crop{i}.jpg'
        result = ocr.ocr(img_path, cls=True)
        kt1=time.time()
        print("thoi gian doc bien : ",kt1-bd1)
        print('result :',len(result))
        if len(result) ==1:
            print('Bien so xe la :', result[0][1][0])
            cv.putText(frame, result[0][1][0], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        if len(result) == 2:
            print('Bien so xe la :',result[0][1][0],result[1][1][0])
            cv.putText(frame, f'{result[0][1][0]}.{result[1][1][0]}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        if len(result) ==[]:
           label= 'k biet'
           print('Bien so xe la :', label)
           cv.putText(frame,label , (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
        if len(result) >=3 or len(result)==0:
            label = 'k biet'
            print('Bien so xe la :', label)
            cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

        #---------------------------------------------------------------------------------------
    # cv.imwrite('./Anh/OCR.jpg',frame)
    cv.imshow('kk', frame)
    cv.waitKey(0)

    return frame

if __name__ == '__main__':
    frame = cv.imread('./Anh/test2.jpg')

    detec(frame)
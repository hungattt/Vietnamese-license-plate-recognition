import cv2 as cv
from imutils import perspective
from paddleocr import PaddleOCR
import numpy as np
import cv2
import time


cap = cv2.VideoCapture('./video/test.MOV')
prev_frame_time = 0
new_frame_time = 0
dem=0

net = cv.dnn_DetectionModel('./setup/yolov4-custom.cfg',
                                './setup/yolov4-custom_best_dot2.weights')
net.setInputSize(416, 416)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

while (cap.isOpened()):

    ret, frame = cap.read()
    dem=dem+1
    if not ret:
        break

    if dem % 1==0:

        classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
        # print(boxes)
        # print(len(boxes))
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            # print(boxes[i])
            poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            # print(poly)
            poly = np.array(poly).astype(int)
            LpRegion = perspective.four_point_transform(frame, poly)
            cv.imwrite(f'./Anh/crop{i}.jpg', LpRegion)
            frame = cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
            frame = cv.circle(frame, (x + w, y + h), 3, (0, 0, 255), -1)
            cv.rectangle(frame, boxes[i], color=(0, 255, 0), thickness=2)
            print(200 * '-')
            # --------------------------------doc bien so--------------------------------------
            ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
            img_path = f'./Anh/crop{i}.jpg'
            result = ocr.ocr(img_path, cls=True)
            print('result :', len(result))
            if len(result) == 1:
                print('Bien so xe la :', result[0][1][0])
                cv.putText(frame, result[0][1][0], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if len(result) == 2:
                print('Bien so xe la :', result[0][1][0], result[1][1][0])
                cv.putText(frame, f'{result[0][1][0]}.{result[1][1][0]}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                           2)
            if len(result) == []:
                label = 'k biet'
                print('Bien so xe la :', label)
                cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if len(result) >= 3 or len(result) == 0:
                label = 'k biet'
                print('Bien so xe la :', label)
                cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ---------------------------------------------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    print("new_frame_time - prev_frame_time : ",new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv.imshow('kk', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()
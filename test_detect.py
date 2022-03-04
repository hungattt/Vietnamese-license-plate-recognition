import cv2 as cv
from imutils import perspective
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
def detec(frame):

    net = cv.dnn_DetectionModel('./setup/yolov4-custom.cfg',
                                './setup/yolov4-custom_best_dot2.weights')
    net.setInputSize(416, 416)

    net.setInputScale(1.0 / 255)

    net.setInputSwapRB(True)

    # frame = cv.imread('/content/bien_so_oto_dep_3.jpg')

    with open('./setup/obj.names', 'rt') as f:
        names = f.read().rstrip('\n').split('\n')

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    print(boxes)
    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
        label = '%.2f' % confidence
        label = '%s: %s' % (names[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y, w, h = box
        print(box)
        #---------------------------crop bien so-----------------------------
        poly=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
        print(poly)
        poly = np.array(poly).astype(int)
        LpRegion = perspective.four_point_transform(frame, poly)
        #-------------------------------------------------------------------

        frame=cv.circle(frame, (x, y), 3, (0, 0, 255), -1)
        frame=cv.circle(frame, (x+w, y+h), 3, (0, 0, 255), -1)
        cv.rectangle(frame, box, color=(0, 255, 0), thickness=2)
        # cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), 2)
        # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv.imwrite('./Anh/crop01.jpg',LpRegion)
        #----------------------------------------------------------------------
        ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
        img_path = './Anh/crop01.jpg'
        result = ocr.ocr(img_path, cls=True)

        print('Bien so xe la :', result[0][1][0])
        #--------------------------------------------------------------------------------
        cv.putText(frame, result[0][1][0], (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv.imshow('crop', LpRegion)
        cv.imshow('kk',frame)
        cv.waitKey(0)


    return frame

if __name__ == '__main__':
    frame = cv.imread('./Anh/testbs5.jpg')
    detec(frame)
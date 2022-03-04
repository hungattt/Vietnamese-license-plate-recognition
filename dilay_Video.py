# import cv2
#
# cap = cv2.VideoCapture('./video/test.MOV')
# frameTime = 100 # time of each frame in ms, you can add logic to change this value.
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(frameTime) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# import cv2
#
# cap = cv2.VideoCapture('./video/test.MOV')
# i=0 #frame counter
# frameTime = 300 # time of each frame in ms, you can add logic to change this value.
# while(cap.isOpened()):
#     ret = cap.grab() #grab frame
#     i=i+1 #increment counter
#     if i % 3 == 0: # display only one third of the frames, you can change this parameter according to your needs
#         ret, frame = cap.retrieve() #decode frame
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(frameTime) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()


#---------------------------FPS---------------------------
import numpy as np
import cv2
import time


cap = cv2.VideoCapture('./video/test.MOV')

# được sử dụng để ghi lại thời gian khi chúng tôi xử lý khung hình cuối cùng
prev_frame_time = 0

# được sử dụng để ghi lại thời gian mà chúng tôi xử lý khung hiện tại
new_frame_time = 0

# Reading the video file until finished
while (cap.isOpened()):

    # Capture frame-by-frame

    ret, frame = cap.read()

    # if video finished or no Video Input
    if not ret:
        break

    # Our operations on the frame come here
    gray = frame

    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # thời gian khi chúng tôi xử lý xong khung này
    new_frame_time = time.time()

    # Tính toán khung hình / giây

    # fps sẽ là số khung hình được xử lý trong khung thời gian nhất định
    # vì phần lớn thời gian của chúng sẽ là sai số 0,001 giây
    # chúng tôi sẽ trừ nó để có kết quả chính xác hơn
    fps = 1 / (new_frame_time - prev_frame_time)
    print("new_frame_time - prev_frame_time :",new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # displaying the frame with fps
    cv2.imshow('frame', gray)

    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()
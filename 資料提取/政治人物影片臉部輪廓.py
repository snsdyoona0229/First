from __future__ import division
import cv2
import dlib
import numpy as np
import os

os.makedirs('C:\\Users\\admin\\Desktop\\2000') #建文件夾，用於保存原始影像中截取的每一幀
os.makedirs('C:\\Users\\admin\\Desktop\\2001') # 建文件夾，用於保存描繪有人臉特徵的圖片
DOWNSAMPLE_RATIO = 4 
photo_number = 400 
video_path = "C:\\Users\\admin\\Desktop\\programing\\final.mp4" # 用於訓練的含有人臉的影像路徑
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))

def prepare_training_data():
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read() # 讀取影像每一幀
        frame_resize = cv2.resize(frame, (0,0), fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1) # 辨識人臉位置
        black_image = np.zeros(frame.shape, np.uint8) # 一張黑色圖片用於描繪人臉特徵
        if len(faces) == 1:
            for face in faces:
                detected_landmarks = predictor(gray, face).parts() # 提取人臉特徵
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

                jaw = reshape_for_polyline(landmarks[0:17])
                left_eyebrow = reshape_for_polyline(landmarks[22:27])
                right_eyebrow = reshape_for_polyline(landmarks[17:22])
                nose_bridge = reshape_for_polyline(landmarks[27:31])
                lower_nose = reshape_for_polyline(landmarks[30:35])
                left_eye = reshape_for_polyline(landmarks[42:48])
                right_eye = reshape_for_polyline(landmarks[36:42])
                outer_lip = reshape_for_polyline(landmarks[48:60])
                inner_lip = reshape_for_polyline(landmarks[60:68])

                color = (255, 255, 255) # 人脸特徵用於白色描繪
                thickness = 3 # 描繪線條粗細

                cv2.polylines(img=black_image, 
                              pts=[jaw,left_eyebrow, right_eyebrow, nose_bridge],
                              isClosed=False,
                              color=color,
                              thickness=thickness)
                cv2.polylines(img=black_image, 
                              pts=[lower_nose, left_eye, right_eye, outer_lip,inner_lip],
                              isClosed=True,
                              color=color,
                              thickness=thickness)

            # 保存圖片
            cv2.imwrite("C:\\Users\\admin\\Desktop\\2000\\original{}.png".format(count), frame)
            cv2.imwrite("C:\\Users\\admin\\Desktop\\2001\\landmarks{}.png".format(count), black_image)
            count += 1
        cv2.imshow('frame', frame)
        cv2.imshow('black_image',black_image)
    # 按下 q 鍵離開迴圈
        if cv2.waitKey(1) == ord('q'):
            break

# 釋放該攝影機裝置
    cap.release()
    cv2.destroyAllWindows()

prepare_training_data()
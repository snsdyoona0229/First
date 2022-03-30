from __future__ import division#舊版本python與新版本相容
import cv2#匯入OpenCV該程式庫，是一個跨平台的影像函式庫
import dlib# Dlib是一套使用C++語言所編寫的函式庫，主要可應用在機器學習、影像處理，以及影像辨識等等
import numpy as np#Numpy 是許多 Python 資料科學套件的基礎，讓使用者可以很容易建立向量（Vector）、矩陣（Matrix）等進行高效率的大量資料運算。
import os# OS模組，這個是調用操作系統命令，來達成建立文件，刪除文件，查詢文件等

os.makedirs('C:\\Users\\admin\\Desktop\\2000') #建文件夾，用於保存原始影像中截取的每一幀
os.makedirs('C:\\Users\\admin\\Desktop\\2001') # 建文件夾，用於保存描繪有人臉特徵的圖片
DOWNSAMPLE_RATIO = 4 # 圖片縮小比例，小圖片加快人臉檢測與特徵提取速度
photo_number = 2000 # 從影像中提取2000張含有人臉特徵的幀
video_path = 0# 用於訓練的含有人臉的影像路徑
detector = dlib.get_frontal_face_detector()# Dlib 的人臉偵測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#在捕捉的臉部預測臉部landmarks。函數參數shape_predictor_68_face_landmarks.dat機器學習育訓練好的模型。

def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))
	

def prepare_training_data():
    cap = cv2.VideoCapture(video_path)#準備擷取攝影機的影像之前，要先呼叫 cv2.VideoCapture 建立一個 VideoCapture 物件
    count = 0
    while cap.isOpened():#cap.isOpened() 檢查攝影機是否有啟動，若沒有啟動則呼叫 cap.open() 啟動它
        ret, frame = cap.read()  # 讀取影像每一幀，建立好 VideoCapture 物件之後，就可以使用它的 read 函數來擷取一張張連續的畫面影像了，以下是一個即時擷取與顯示畫面
        frame_resize = cv2.resize(frame, (0,0), fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)#調整圖像大小原始比例0.25
        gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)#透過函式轉為灰階
        faces = detector(gray, 1) # 檢測灰度影象中的臉部
        black_image = np.zeros(frame.shape, np.uint8) # 一張黑色圖片用於描繪人臉特徵

        if len(faces)==1:#有偵測到人臉
            for face in faces:
                detected_landmarks = predictor(gray, face).parts() # 提取人臉特徵
                landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

                jaw = reshape_for_polyline(landmarks[0:17])#人臉區域
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

                cv2.polylines(img=black_image, #Python 來畫多邊形 polylines
                              pts=[jaw,left_eyebrow, right_eyebrow, nose_bridge],#cv2.polylines(影像, 頂點座標, 封閉型, 顏色, 線條寬度)
                              isClosed=False,#封閉型參數是一個布林值，若設定為 True 的話，它就會自動把最後一個點座標跟第一個點座標連起來，反之就是不連這一條線段
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


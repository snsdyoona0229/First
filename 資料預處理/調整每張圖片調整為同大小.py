import os
import cv2
import numpy as np

def resize_img(DATADIR, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    #設置圖片大小256
    path = os.path.join(DATADIR, data_k)
    #cv2.imread讀入圖片，讀入格式為IMREAD_COLOR
    img_licv2.imread讀入圖片，讀入格式為IMREAD_COLORst = os.listdir(path)
 
    for i in img_list:
        if i.endswith('.png'):
            # cv2.imread讀入圖片，讀入格式為IMREAD_COLOR
            img_array = cv2.imread((path + '\\' + i), cv2.IMREAD_COLOR)
            # cv2.resize函數resize圖片
            new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
            img_name = str(i)
            '''生成圖片存儲的目標路徑'''
            save_path = 'C:\\Users\\admin\\Desktop\\aa1\\FACEB\\10\\'
            if os.path.exists(save_path):
                print(i)
                '''cv.2的imwrite函數存取圖片'''
                save_img=save_path+img_name
                cv2.imwrite(save_img, new_array)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)
 
 
if __name__ == '__main__':
    #圖片路径
    DATADIR = "C:\\Users\\admin\\Desktop\\aa1"
    data_k = 'FACEB'
    #需要修改的新的尺寸
    img_size = [256, 256]
    resize_img(DATADIR, data_k, img_size)
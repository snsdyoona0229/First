import os
import cv2
import numpy as np
#img =['0train256original1.png']
#img01 = ['landmarks0.png']

path = 'C:\\Users\\admin\\Desktop\\2000\\new2000\\oface0429256\\'
path2= 'C:\\Users\\admin\\Desktop\\2000\\new2000\\lmark0429256\\'

for i in range(0,780,1) :

    
    img1 = cv2.imread(path + 'original'+str(i)+'.png')
    img2 = cv2.imread(path2 +'landmarks'+str(i)+'.png')
    
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('C:\\Users\\admin\\Desktop\\2000\\new2000\\con0429\\'+str(i)+'.png',vis)
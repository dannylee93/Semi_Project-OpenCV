import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import copy
import random
import collections
from threading import Thread, Lock
import csv





x_list = []
y_list = []
w_list = []
h_list = []



BLUR = 21
CANNY_THRESH_1 = 5
CANNY_THRESH_2 = 10
MASK_DILATE_ITER = 20
MASK_ERODE_ITER = 20
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


raw_path = './data/best/'
raw_list = sorted(os.listdir(raw_path))

save_path = './data/00_edge/'
contour_path = './data/01_contourt'
crop_path = './data/02_crop'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(contour_path):
    os.mkdir(contour_path)
if not os.path.exists(crop_path):
    os.mkdir(crop_path)


for i in raw_list : 
    img_path = os.path.join(raw_path, i)
    #print (i)
    img_orig = cv2.imread(img_path)
 #   img_orig =  img_orig.astype('uint8')
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    edge = cv2.Canny(img_blur, 50, 120)
    cv2.imwrite(os.path.join(save_path, i), edge)
    _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCnt = 0
    new_contours = []
    for c in contours :
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
        if len(approx) == 4:  # Select the contour with 4 corners           
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            new_contours.append(c)
            cv2.drawContours(img_orig, [NumberPlateCnt], -1, (0,255,0), 3)
            cv2.imwrite(os.path.join(contour_path, i), img_orig)
            break

    for item in new_contours :
        x, y, w, h = cv2.boundingRect(c)
        print (type(x))
        with open('xywh.csv', 'w') as df:
            write = csv.writer(df, delimiter = ',')
            write.writerow(x)
                
 
        
        
'''
        xmin = int(np.round(item[0] * width))
        ymin = int(np.round(item[1] * height))
        xmax = int(np.round(item[2] * width))
        ymax = int(np.round(item[3] * height))

'''



        
#        x, y, w, h = cv2.boundingRect(c)
#        print (len(x))
        
        


            # save to disk

            
            #print(NumberPlateCnt[5])

            #print(NumberPlateCnt.shape)

            #for z in len(


 
 #           break

'''
        x_min, x_max = 0, 0
        value = []

        for a in range(len(contour_xy)):
            for z in range(len(contour_xy[a])):
                value.append(contour_xy[a][z][0][0])
                x_min = min(value)
                x_max = max(value)
        print ('xmin:', x_min, 'xmax:', x_max)
'''
            



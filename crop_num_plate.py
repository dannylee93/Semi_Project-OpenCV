#-*- coding:utf-8 -*-
import numpy as np
import cv2
import os
from PIL import Image 
import pytesseract
import csv
img_list = []
number_list = []
error_list = []

raw_path = './data/best/'
raw_list = sorted(os.listdir(raw_path))

save_path = './data/00_edge/'
contour_path = './data/01_contourt/'
crop_path = './data/02_crop/'

if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(contour_path):
    os.mkdir(contour_path)
if not os.path.exists(crop_path):
    os.mkdir(crop_path)

count = 0
for i in raw_list : 
    img_path = os.path.join(raw_path, i)
    img_orig = cv2.imread(img_path)
    img_orig_copy = img_orig.copy()
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    edge = cv2.Canny(img_blur, 50, 120)
    cv2.imwrite(save_path + 'ed_%04d.jpg' % (count), edge)
    _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
    new_contours = []
    for c in contours :
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:  # Select the contour with 4 corners           
            cntcnt = approx 
            cv2.drawContours(img_orig, [cntcnt], -1, (0,255,0), 3)
            new_contours.append(c)
            cv2.imwrite(contour_path + 'cnt_%04d.jpg' % (count), img_orig)


    for item in new_contours :
        x, y, w, h = cv2.boundingRect(item)
        #print (x, y, w, h)
        ROI = img_orig_copy[y:y+h, x:x+w]
        cv2.imwrite(crop_path + 'crop_%04d.jpg' % (count), ROI)

        count += 1

        break

crop_list = sorted(os.listdir(crop_path))
for q in crop_list :
    crop_img = os.path.join(crop_path, q)
    crop_orig = cv2.imread(crop_img)
    
    img_list.append(q)
    number = pytesseract.image_to_string(crop_orig, lang = 'eng')
    
    if number == '':
        error_list.append('FAILED')    
    else:
        number_list.append(number)
        
    
    with open('result_num_plate.csv', 'w') as df:
        write = csv.writer(df, delimiter = ',')
        
        write.writerows([img_list])
        write.writerows([error_list])
        write.writerows([number_list])
        
        
 

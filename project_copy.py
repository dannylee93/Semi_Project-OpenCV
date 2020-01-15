import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


raw_path = './data/best/'
raw_list = sorted(os.listdir(raw_path))

save_path = './data/edge/'
contour_path = './data/contourt'
crop_path = './data/crop'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(contour_path):
    os.mkdir(contour_path)
if not os.path.exists(crop_path):
    os.mkdir(crop_path)


for i in raw_list : 
    img_path = os.path.join(raw_path, i)
    img_orig = cv2.imread(img_path)
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    edge = cv2.Canny(img_blur, 50, 120)
    cv2.imwrite(os.path.join(save_path, i), edge)
    contour_info = []
    _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCnt = 0
    for c in contours :
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #print (len(approx))
                
        
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            #print (NumberPlateCnt)
            
            cv2.drawContours(img_orig, [NumberPlateCnt], -1, (0,255,0), 3)
            cv2.imwrite(os.path.join(contour_path, i), img_orig)

            break

 

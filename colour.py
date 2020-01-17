import os
import cv2
import numpy as np
import csv


raw_path = './data/best/'
raw_list = sorted(os.listdir(raw_path))
'''
save_path = './data/00_edge/'
contour_path = './data/01_contourt'
crop_path = './data/02_crop'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(contour_path):
    os.mkdir(contour_path)
if not os.path.exists(crop_path):
    os.mkdir(crop_path)
'''
lower_orange = (100, 200, 200)
upper_orange = (140, 255, 255)

lower_green = (30, 80, 80)
upper_green = (70, 255, 255)

lower_blue = (0, 180, 55)
upper_blue = (20, 255, 200)


for i in raw_list : 
    img_path = os.path.join(raw_path, i)
    img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img_result = cv2.bitwise_and(img_orig, img_orig, mask=img_mask)


    cv2.imshow("orange", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

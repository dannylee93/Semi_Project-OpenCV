# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

plt.style.use('dark_background')


img_orig = cv2.imread('./data/test.jpg')
height, width, channel = img_orig.shape

gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./data/test/00_gray.jpg', gray)

img_blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
cv2.imwrite('./data/test/01_blur.jpg', img_blur)

edge = cv2.Canny(img_blur, 170, 150)
cv2.imwrite('./data/test/02_edge.jpg', edge)


# img_thresh = cv2.adaptiveThreshold(img_blur, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)
# cv2.imwrite('./data/test/02_tresh.jpg', img_thresh)
_, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None
cnt = 0
for c in contours :
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx #This is our approx Number Plate Contour
        break




# Drawing the selected contour on the original image
cv2.drawContours(img_orig, [NumberPlateCnt], -1, (0,255,0), 3)
#cv2.imshow("Final Image With Number Plate Detected", img_orig)
cv2.imwrite('./data/test/03_contour.jpg', img_orig)

'''
contours_xy = np.array(contours)
contours_xy.shape

# x의 min과 max 찾기
x_min, x_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
print(x_min)
print(x_max)

# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)
print(y_min)
print(y_max)

# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min

img_trim = img_orig[y:y+h, x:x+w]
cv2.imwrite('org_trim.jpg', img_trim)
#
'''

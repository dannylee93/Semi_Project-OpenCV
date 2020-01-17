

# Semi_Project-OpenCV

> WE Use Open CV to build Codes for Recognizing Car number Plates , And Github Branch(Collaborator) to improve the efficiency of the team project.



## Branch

> *WE REFERRED to Atlassian.com [‘Getting Git Right’ Tutorial - Comparing Workflows]*

<img src="https://wac-cdn.atlassian.com/dam/jcr:0869c664-5bc1-4bf2-bef0-12f3814b3187/01.svg?cdnVersion=485" style="zoom:67%;" />

- `Master branch` :  **@dannylee93** 이동규
- `branch A(brchA)` : **@WinterBlue16**  이경희
- `branch B(brchB)` : **@dannylee93** 이동규
- `branch C(brchC)` : **@imooncoco**  이가은
- `branch D(brchD)` : **@siraasony** 이소은
- `branch E(brchE)` : **@HanSangKook**  한상국
- `branch E(brchE)` : **@tkxkd0159**  이재승



## Basic Concept

Use the features of Open CV to detect the number of Car license plates of a car and compare the images to the test set to see how accurate they are.

###  Our Goal

- **Accuracy**: For various styles of vehicle images, the vehicle license plate is recognized accurately and the code is created to allow the number to be extracted.

- **Efficiency**: Refer to the Atlassian's co-project method and workflow of Github to increase the efficiency of the project progress.

- **Find New Way**: Find new ways to make ideas such as Chatbot API and github collaborator come true and try.

  

###  OverView

**Feature Branch Workflow**: We refer to this workflow to facilitate the communication of team members to produce collaborative results. The **key concept** is that each person makes a brandy and works. This workflow provides an isolated work environment so that team members can safely develop new features around the main code.

<img src="https://wac-cdn.atlassian.com/dam/jcr:a9cea7b7-23c3-41a7-a4e0-affa053d9ea7/04%20(1).svg?cdnVersion=485" style="zoom:80%;" />

> The image shows the most advanced version of the workflow we refer to.

**recogizing Car number plates**: It is detected through matching the previously extracted license plate image using the `BFmatcher with KNN Match` of the Open CV.

<img src="C:\Users\bruce0809\Desktop\matcher.jpg" style="zoom: 67%;" />



### View Codes

A *View* Car license plates data. 

To detect the Car license plate image in the test set, thresholding was performed using Gaussian Blur, and Contour was extracted using the characteristics of the vehicle license plate.

```python
# 번호판 이미지 검출
img_cropped = cv2.getRectSubPix(
    img_rotated,patchSize = (int(plate_width),int(plate_height)), 
                                center=(int(plate_cx), int(plate_cy))
	)
```

During the process of building codes related to image detection, an error was found due to the inability of the extractor to find a matching point between the license plate and the vehicle image, although the search was conducted. Change codes with ORB + KNN + FLANNMatcher to **SURF + BFMatcher**  codes.

```python
detector = cv2.xfeatures2d.SURF_create()
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

​```
##비교할 디렉토리의 객체 정의 중 일부...

matches = matcher.match(desc1, desc2)
# 매칭 결과를 거리기준 오름차순으로 정렬 ---③
matches = sorted(matches, key=lambda x:x.distance)

# 최소 거리 값과 최대 거리 값 확보 ---④
min_dist, max_dist = matches[0].distance, matches[-1].distance
# 최소 거리의 20% 지점을 임계점으로 설정 ---⑤
ratio = 0.2
good_thresh = (max_dist - min_dist) * ratio + min_dist

# 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
#m.distance 매칭객체의 거리 공통함수부분참조  
good_matches = [m for m in matches if m.distance < good_thresh]
```



## Requirements

- **Python** 3.7.3
- **Open CV** 3.4.2.16
- **Tesseract** 3.05



## Examples

- [Book](https://github.com/ReactorKit/ReactorKit/tree/master/Examples/Counter): 파이썬을 활용한 Open CV 프로젝트

- [GitHub Search](https://github.com/ReactorKit/ReactorKit/tree/master/Examples/GitHubSearch): 빵형의 개발도상국(https://github.com/kairess/license_plate_recognition/)

  

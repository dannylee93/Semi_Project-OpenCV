import cv2, glob, numpy as np
detector = cv2.ORB_create()
FLANN_INDEX_LSH = 6
index_params = {'algorithm' : FLANN_INDEX_LSH, 'table_number' : 6, 'key_size' : 12, 'multi_probe_level' : 1}
search_params = {'checks': 32}
matcher = cv2.FlannBasedMatcher(index_params, search_params)

def serch(img):
 #   gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(img, None)

    results = {}

    img_paths = glob.glob('./img/*.*')
    for img_path in img_paths:
        cars = cv2.imread(img_path)
        #print("======================",cars.shape)
        cv2.imshow('searching..', cars)
        cv2.waitKey(5)

        gray2 = cv2.cvtColor(cars, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        matches = matcher.knnMatch(desc1, desc2, 2)
        #print("======================matches:",matches)

        ratio = 0.7
        good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance*ratio]
        #print("======================good_matches:",good_matches)
        

        MIN_MATCH = 10
        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            accuracy = float(mask.sum()) / mask.size
            #print("======================",accuracy)

            results[img_path] = accuracy
 #       cv2.destroyAllWindows('serching..')

    if len(results) > 0 :
        results = sorted([(v, k) for (k, v) in results.items() if v>0], reverse=True)
        #print("======================",results.shape)
        return results

img_test = cv2.imread('img_test.jpg')
gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
results = serch(gray)
#print("======================results:",results)


if len(results) == 0 :
    print("NO matched cars found")
else:
    for(i, (accuracy, img_path)) in enumerate(results):
        print(i, img_path, accuracy)
        if i == 0:
            cars = cv2.imread(img_path)
            cv2.putText(cars, ("Accuracy:%.2f%%"%(accuracy*100)), (10,100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('result', cars)
cv2.waitKey()
cv2.destroyAllWindows()



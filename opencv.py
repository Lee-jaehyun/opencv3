import cv2
import numpy as np
import os


path= os.path.dirname(os.path.abspath(__file__))
print(path)


print(cv2.__version__)

cv2.TrackerKCF

face_cascade = cv2.CascadeClassifier("haarcascade_frontface.xml")

image = cv2.imread("../../../Desktop/cnn/lenna.png")
image = cv2.resize(image, dsize=(1200, 1200))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)

faces = face_cascade.detectMultiScale(gray, 1.1, 2)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 눈 찾기
    roi_color = image[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]

# 영상 출력
cv2.imshow('image', image)

cv2.waitKey(0)
print(faces)
## face output 은 [[x,y,w,h]] 순서로 얻어진다.

cv2.imshow("face", image[faces[0][1]:faces[0][1]+faces[0][3], :])
cv2.waitKey(0)
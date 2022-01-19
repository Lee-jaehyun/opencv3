import numpy as np
import cv2
from cdf import CDF



#image = cv2.imread("../../../Desktop/cnn/lenna.png")


image = cv2.imread("../../../Desktop/test_data_set2/IMG_1396.jpg")

image = cv2.resize(image, dsize=(800, 800))

ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)     # OpenCV의 경우 YCrCb 순서로 변환된다.

y2 = ycbcr[:,:,0]
cr2 = ycbcr[:,:,1]
cb2 = ycbcr[:,:,2]


lower = np.array([0, 133, 77])
upper = np.array([255, 173, 127])


## ycbcr 성분에서 lower , upper 사이의 값을 추출하는 방법
skin = cv2.inRange(ycbcr, lower, upper)
skin2 = cv2.inRange(ycbcr, lower, upper)

result = cv2.bitwise_and(image, image, mask=skin2)    # 이미지 비트 연산 원하는 영역을 흰색으로 보여지게 함. (mask 범위 내의 두 이미지에 대해)


print("Y_Shape :",y2.shape)
print("YCBCR_Shape: ",ycbcr.shape)
print("CR_Shape: ",cr2.shape)


#print(len(cr2[0]))
#dst = cv2.add(cr2, cb2)

#dst = cv2.subtract(cr2, cb2)


cv2.imshow("Y", y2)
cv2.imshow("cr", cr2)
cv2.imshow("cb", cb2)
cv2.imshow("skin", skin)
cv2.imshow("skin2", result)
#cv2.imshow("ycbcr", dst)
cv2.waitKey(0)

print(y2.shape)


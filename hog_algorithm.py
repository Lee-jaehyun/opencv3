import cv2

hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

src = cv2.imread("/Users/jaehyuni/Desktop/바이오커넥트/상명대/people.jpeg")

detected, _ = hog.detectMultiScale(src)

for (x, y, w, h) in detected:
    cv2.rectangle(src, (x,y), (x+w, y+h), (50, 200, 50), 3)

cv2.imshow("people", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
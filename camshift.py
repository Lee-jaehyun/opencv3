import cv2
import sys

cap = cv2.VideoCapture('/Users/jaehyuni/Downloads/IMG_1485.MOV')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 초기 사각형 영역 : (x, y, w, h)
x, y, w, h = 300, 300, 100, 100
rc = (x, y, w, h)

ret, frame = cap.read()

if not ret:
    print('frame read failed!')
    sys.exit()

roi = frame[y:y + h, x:x + w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# HS 히스토그램 계산
channels = [0, 1]
ranges = [0, 180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90, 128], ranges)

# CamShift 알고리즘 종료 기준
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # HS 히스토그램에 대한 역투영
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    # CamShift
    ret, rc = cv2.CamShift(backproj, rc, term_crit)

    # 추적 결과 화면 출력
    cv2.rectangle(frame, rc, (0, 0, 255), 2)
    # 타원 그리는 함수, 사각형 좌표 주어지면 사각형에 내접하는 타원을 그려줌
    cv2.ellipse(frame, ret, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 27:
        break

cap.release()
cv2.destroyAllWindows()




## 출처 : http://www.gisdeveloper.co.kr/?p=6838
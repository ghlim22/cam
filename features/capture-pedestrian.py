import cv2
import os
from datetime import datetime
import time

def captureHuman(hog: cv2.HOGDescriptor, timeInterval: float) -> bool:
    found, _ = hog.detectMultiScale(img)
    for (x, y, w, h) in found:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255))
        found_roi = img[y:y+h, x:x+w]
        if timeInterval > 10 :
            if not os.path.exists('detections/' + datetime.now().strftime('%Y-%m-%d')):
                os.mkdir('detections/' + datetime.now().strftime("%Y-%m-%d"))
            cv2.imwrite('detections/' + datetime.now().strftime("%Y-%m-%d") + '/' + datetime.now().strftime("%H-%M") + '.jpg', found_roi)
            return True
    return False

# Main

if not os.path.exists('detections'):
	os.mkdir('detections')

hogdef = cv2.HOGDescriptor()
hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("img/small.mp4")
mode = True
print("Toggle space bar to change mode.")
if cap.isOpened():
    print('Success!')
ts1 = time.time()
while cap.isOpened():
    ts2 = time.time()
    ret, img = cap.read()
    if captureHuman(hogdef, ts2 - ts1):
        ts1 = ts2
    cv2.imshow('frame', img)
    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows
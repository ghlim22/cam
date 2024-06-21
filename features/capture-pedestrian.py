import cv2
import os
from datetime import datetime
import time

INTERVAL = 2
SIZE = (640, 360)


def captureHuman(
    hog: cv2.HOGDescriptor, timeInterval: float, img: cv2.typing.MatLike
) -> None:
    found, _ = hog.detectMultiScale(img)
    for x, y, w, h in found:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255))
        found_roi = img[y : y + h, x : x + w]
        if timeInterval > INTERVAL:
            if not os.path.exists("detections/" + datetime.now().strftime("%Y-%m-%d")):
                os.mkdir("detections/" + datetime.now().strftime("%Y-%m-%d"))
            cv2.imwrite(
                "detections/"
                + datetime.now().strftime("%Y-%m-%d")
                + "/"
                + datetime.now().strftime("%H-%M-%S")
                + ".jpg",
                found_roi,
            )


# Main

if not os.path.exists("detections"):
    os.mkdir("detections")

hogdef = cv2.HOGDescriptor()
hogdef.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("img/sample8.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter.fourcc(*"DIVX")
out = cv2.VideoWriter("output.avi", fourcc, fps, SIZE)

ts1 = time.time()

while cap.isOpened():
    ts2 = time.time()
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, SIZE)
    captureHuman(hogdef, ts2 - ts1, img)
    if ts2 - ts1 > INTERVAL:
        ts1 = ts2
    cv2.imshow("frame", img)
    out.write(img)
    if cv2.waitKey(3) == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows

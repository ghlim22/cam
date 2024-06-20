import cv2
import os
import time
from datetime import datetime

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

fonts = cv2.FONT_HERSHEY_COMPLEX

face_detector = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")


# returns the face width in the pixels
def get_face_width(img: cv2.typing.MatLike) -> int:
    face_width = 0
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)
    for x, y, h, w in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        face_width = w
    return face_width


def get_face_data(img: cv2.typing.MatLike) -> tuple[int, int, int, int, int]:
    face_width = 0
    face_x = 0
    face_y = 0
    height = 0
    width = 0
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)
    for x, y, h, w in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        face_width = w
        face_x = x
        face_y = y
        height = h
        width = w
    return face_width, face_x, face_y, height, width


def get_focal_length(
    measured_distance: float, real_width: float, width_in_rf_image: int
) -> float:
    len = (width_in_rf_image * measured_distance) / real_width
    return len


def get_distance(
    focal_length: float, real_face_width: float, face_width_in_frame: float
) -> float:
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


def capture_face(img: cv2.typing.MatLike, x: int, y: int, h: int, w: int) -> None:
    face = img[y : y + h, x : x + w]
    if not os.path.exists("detections/" + datetime.now().strftime("%Y-%m-%d")):
        os.mkdir("detections/" + datetime.now().strftime("%Y-%m-%d"))
    cv2.imwrite(
        "detections/"
        + datetime.now().strftime("%Y-%m-%d")
        + "/"
        + datetime.now().strftime("%H-%M")
        + ".jpg",
        face,
    )


# reference image data
known_distance = 65.0
known_width = 14.3
focal_length_found = 981.81

video_path = "./img/sample1.mp4"
ts1 = time.time()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter.fourcc("D", "I", "V", "X")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
output_path = "./out.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (640, 380))

while cap.isOpened():
    ts2 = time.time()
    (success, frame) = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, (640, 480))
    face_width_in_frame, x, y, h, w = get_face_data(frame)
    if face_width_in_frame != 0:
        distance = get_distance(focal_length_found, known_width, face_width_in_frame)
        # cv2.line(frame, (30, 30), (230, 30), RED, 32)
        # cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
        cv2.putText(
            frame,
            f"Distance: {round(distance, 2)} cm",
            (x, y - 10),
            fonts,
            0.6,
            GREEN,
            2,
        )
        if ts2 - ts1 > 10.0:
            ts1 = ts2
            capture_face(frame, x, y, h, w)
    cv2.imshow("frame", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break
out.release()
cap.release()
cv2.destroyAllWindows()

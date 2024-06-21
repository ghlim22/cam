import cv2
import random
import sys
from queue import Queue

GREEN = (0, 255, 0)
WIDTH = 640
HEIGHT = 360
SIZE = (WIDTH, HEIGHT)


def select_roi_manually(
    frame: cv2.typing.MatLike,
    lst: list[cv2.typing.Rect],
    color_list: list[tuple[int, int, int]],
) -> None:
    frame = cv2.resize(frame, SIZE)
    while True:
        box: cv2.typing.Rect = cv2.selectROI("object-tracking", frame)
        if box[0] == 0:
            break
        lst.append(box)
        color_list.append(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )


def select_roi(
    frame: cv2.typing.MatLike,
    box_list: list[cv2.typing.Rect],
    color_list: list[tuple[int, int, int]],
    flag: bool,
) -> None:
    if flag:
        hog: cv2.HOGDescriptor = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
    else:
        hog = cv2.HOGDescriptor((48, 96), (16, 16), (8, 8), (8, 8), 9)
        hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
    frame = cv2.resize(frame, SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    found_list, _ = hog.detectMultiScale(frame)
    for obj in found_list:
        box_list.append(obj)
        color_list.append(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )


# main
if len(sys.argv) < 2:
    print("put video file path as first argument")
    sys.exit(0)
video_path: str = sys.argv[1]
cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
ret: tuple[bool, cv2.typing.MatLike] = cap.read()
success: bool = ret[0]
frame = ret[1]
if not success:
    print("Failed to read video")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)


boxes: list[cv2.typing.Rect] = []
colors: list[tuple[int, int, int]] = []
size_sum: list[int] = []
size_list: list[Queue[int]] = []

# select_roi(frame, boxes, colors, False)
fourcc = cv2.VideoWriter.fourcc(*"DIVX")
out = cv2.VideoWriter("output.avi", fourcc, fps, SIZE)

select_roi_manually(frame, boxes, colors)

multi_tracker = cv2.legacy.MultiTracker.create()

for box in boxes:
    multi_tracker.add(cv2.legacy.TrackerCSRT.create(), frame, box)
    size_sum.append(0)
    size_list.append(Queue())

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.resize(frame, SIZE)
    success, boxes = multi_tracker.update(frame)
    for i, box in enumerate(boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        cv2.putText(
            frame,
            "Number: %d" % i,
            (int(box[0]) - 10, int(box[1]) - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.4,
            (0, 10, 245),
        )
        size = int(box[2]) * int(box[3])
        cv2.putText(
            frame,
            "Size: %d" % size,
            (int(box[0]) - 10, int(box[1])),
            cv2.FONT_HERSHEY_DUPLEX,
            0.4,
            (255, 0, 0),
        )
        size_list[i].put(size)
        if size_list[i].qsize() > 10:
            size_sum[i] -= size_list[i].get()
        size_sum[i] += size
        if size > 900 and size > (size_sum[i] / size_list[i].qsize() + 1500):
            cv2.putText(
                frame,
                "Approaching",
                (int(box[0]) - 10, int(box[1]) + 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 0, 0),
            )

    cv2.imshow("tracker", frame)
    out.write(frame)
    if cv2.waitKey(int(1000 / fps)) == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

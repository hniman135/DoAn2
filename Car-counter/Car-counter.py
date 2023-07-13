import numpy as np
from ultralytics import  YOLO
import cv2
import cvzone
import math
from sort import *


# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/cars.mp4")


model = YOLO("../YOLO-weight/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3 , iou_threshold=0.3)

limit = [400, 297, 673, 297]

totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 3)

            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            curClass = classNames[cls]
            if curClass == "car" or curClass == "motorbike"\
                    or curClass == "bus" or curClass == "truck"\
                    and conf>0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                # cvzone.putTextRect(img, f'{curClass} {conf}',
                #                    (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                curArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, curArray))


    resultsTracker = tracker.update(detections)
    cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0,0,255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(Id)}',
                           (max(0, x1), max(35, y1)),
                           scale=0.6, thickness=1, offset=3)

        cx, cy = x1 + w // 2 , y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

        if limit[0]<cx<limit[2] and limit[1]-20<cy<limit[1]+20:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                cv2.line(img, (limit[0], limit[1]), (limit[2], limit[3]), (0,255,0), 5)

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50,50))

    cv2.imshow("image", img)
    #cv2.imshow("imageRegion", imgRegion)
    cv2.waitKey(0)
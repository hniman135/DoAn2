import numpy as np
from read_infor import checker
from license_plate_detect import license_plate_detect
from license_plate_ocr import OCR
from licence_plate_segment import character_segmentation
from sort import *
import cv2
import imutils
from ultralytics import YOLO
import cvzone
import math
from postMessage import *
import pymongo
import time
import random
import string

# plate = "59L206377"
# status, payment = checker(plate)

client = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = client['License_Plate_Manager']
nor_his = mydb.nor_his
vip_his = mydb.vip_his
gen_his = mydb.gen_his

img_path = "C:/Users/vomin/Source/PycharmProjects/object_detection/0000_00532_b.jpg"
def read_plate(img_path):
    lpd = license_plate_detect()
    # ocr_read = OCR()
    seg_read = character_segmentation(n_clusters=3)
    

# load the input image from disk and resize it
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=600)
    # apply automatic license plate recognition
    lpCnt, ROI_list = lpd.crop(image)
    if len(lpCnt) == 0:
        print("Khong tim thay bien so")
        cv2.destroyAllWindows()
        
    for (i, r) in enumerate(ROI_list):
            char = seg_read.segmentation(r)
            if len(char) > 2:
                print('Bien so:\n' + char)
    return char


def main():
    cap = cv2.VideoCapture("./Videos/demo1.mp4")


    model = YOLO("./YOLO-weight/yolov8l.pt")

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
    tracker = Sort(max_age=20, min_hits=3 , iou_threshold=0.3)
    mask = cv2.imread("mask.png")

    nor_line = [840, 200, 1100, 150]
    vip_line = [360, 820, 960, 600]
    check_line = [330, 760, 600, 300]
    total_check = 0
    total_nor = 0
    total_vip = 0
    checkCount = []
    norCount = []
    vipCount = []
    status = 0
    balance = 0
    finish = True
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
                currentClass = classNames[cls]
                if currentClass == "car":
                    # cvzone.cornerRect(img, (x1, y1, w, h))
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                    curArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, curArray))
        resultsTracker = tracker.update(detections)
        cv2.line(img, (nor_line[0], nor_line[1]), (nor_line[2], nor_line[3]), (0, 0, 255), 5)
        cv2.line(img, (vip_line[0], vip_line[1]), (vip_line[2], vip_line[3]), (0, 0, 255), 5)
        cv2.line(img, (check_line[0],check_line[1]), (check_line[2], check_line[3]), (0, 255, 0), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f'{int(Id)}',(max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            sec = time.localtime()
            if check_line[0] < cx < check_line[2] and check_line[3]  < cy < check_line[1] and finish == True :
                finish = False
                if  checkCount.count(Id) == 0:
                    checkCount.append(Id)
                    total_check  += 1
                    path_input = input("Enter image path: ")
                    plate_number = read_plate(path_input)
                    status, balance = checker(plate_number)

            if (checkCount.count(Id) == 1 and status == 0):
                if nor_line[0] < cx < nor_line[2] and nor_line[3] - 5 < cy < nor_line[1] + 5:
                    if  norCount.count(Id) == 0:
                        norCount.append(Id)
                        total_nor  += 1
                        print("dung zone")
                        QR_raw = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                        rec= [{
                            "Time" : sec,
                            "Plate_number" : plate_number,
                            "Zone": "Zone-N",
                            "Cost": "15k",
                            "QR_code": QR_raw
                            }]
                        nor_his.insert_many(rec)
                        gen_his.insert_many(rec)
                        
                        mess =  "Bien so: " + str(plate_number) + ", Ban dang o Zone-N, 15k/slot" + ", QR code:" + QR_raw
                        print("mess:", mess)
                        changeMess(mess)
                        sendMess()
                        finish = True
                        
            elif (checkCount.count(Id) == 1 and status == 1):
                if nor_line[0] < cx < nor_line[2] and nor_line[3] - 5 < cy < nor_line[1] + 5:
                    if  norCount.count(Id) == 0:
                        norCount.append(Id)
                        total_nor  += 1
                        print("sai zone")
                        QR_raw = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                        rec= [{
                            "Time" : sec,
                            "Plate_number" : plate_number,
                            "Zone": "Zone-N",
                            "Cost": "10k",
                            "QR_code": QR_raw
                            }]
                        vip_his.insert_many(rec)
                        gen_his.insert_many(rec)
                        mess =  "Bien so: " + str(plate_number) + ", So tien con lai: " + str(balance) + ", Ban dang o Zone-N, 10k/slot" + ", QR code:" + QR_raw
                        print("mess:", mess)
                        
                        changeMess(mess)
                        sendMess()
                        finish = True

            if (checkCount.count(Id) == 1 and status == 0):
                if vip_line[0] < cx < vip_line[2] and vip_line[3] - 5 < cy < vip_line[1] + 5:
                    if  vipCount.count(Id) == 0:
                        vipCount.append(Id)
                        total_vip  += 1
                        print("sai zone")
                        QR_raw = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                        rec= [{
                            "Time" : sec,
                            "Plate_number" : plate_number,
                            "Zone": "Zone-V",
                            "Cost": "25k",
                            "QR_code": QR_raw
                            }]
                        nor_his.insert_many(rec)
                        gen_his.insert_many(rec)
                        
                        mess =  "Bien so: " + str(plate_number) + ", Ban dang o Zone-V, 25k/slot\n" + ", QR code:" + QR_raw
                        print("mess:", mess)
                        changeMess(mess)
                        sendMess()
                        finish = True

            if (checkCount.count(Id) == 1 and status == 1):
                if vip_line[0] < cx < vip_line[2] and vip_line[3] - 5 < cy < vip_line[1] + 5:
                    if  vipCount.count(Id) == 0:
                        vipCount.append(Id)
                        total_vip  += 1
                        print("dung zone")
                        QR_raw = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
                        rec= [{
                            "Time" : sec,
                            "Plate_number" : plate_number,
                            "Zone": "Zone-V",
                            "Cost": "10k",
                            "QR_code": QR_raw
                            }]
                        vip_his.insert_many(rec)
                        gen_his.insert_many(rec)
                        
                        mess =  "Bien so: " + str(plate_number) + ", So tien con lai: " + str(balance) + ", Ban dang o Zone-N, 10k/slot\n" + ", QR code:" + QR_raw
                        print("mess:", mess)
                        changeMess(mess)
                        sendMess()
                        finish = True


            #         plate_number = read_plate(img_path)
            #         status, balance = checker(plate_number)
            #         print(status)
        cvzone.putTextRect(img, f' Check {total_check}', (50, 50))
        cvzone.putTextRect(img, f' Nor {total_nor}', (350, 50))
        cvzone.putTextRect(img, f' Vip {total_vip}', (650, 50))

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
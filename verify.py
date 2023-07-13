import cv2
from pyzbar.pyzbar import decode
import os
import pymongo
from license_plate_detect import license_plate_detect
from license_plate_ocr import OCR
from licence_plate_segment import character_segmentation
import imutils
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

path_input = input("Enter image path: ")
plate_number = read_plate(path_input)

cap = cv2. VideoCapture (0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("QR", frame)
    k = cv2.waitKey(1)
    if (decode(frame) != []):
        for code in decode (frame):
            qr_dectect = code.data.decode ('utf-8')
        break
cap.release()
cv2.destroyAllWindows()

client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client["License_Plate_Manager"]
gen_his = db.gen_his
result = gen_his.find({'QR_code' : qr_dectect})
apr = 0
rs=gen_his.find({"QR_code":{"$exists":True}})
cnt = gen_his.count_documents({})
for index in range (0 , cnt): 
    if (rs[index]["QR_code"] == qr_dectect):
        apr = 1
if(apr == 1 ):
    for res in result:
        if(plate_number==res["Plate_number"]):
            print("QR code trung khop")
        else:
            print("QR code khong trung khop")

else:
    print("QR code khong ton tai")
    os.system('python verify.py')

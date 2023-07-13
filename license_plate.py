from license_plate_detect import license_plate_detect
from imutils import paths
import argparse
import imutils
import cv2
from license_plate_ocr import OCR
from licence_plate_segment import character_segmentation


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input directory of images")
    ap.add_argument("-d", "--debug", type=int, default=-1,
                    help="whether or not to show additional visualizations")
    ap.add_argument("-s", "--segment", type=int, default=0,
                    help="split license plate segment")
    args = vars(ap.parse_args())

    return args

def main(args):
    cnt = 0
    count_len = 0
    stop = 0
    lpd = license_plate_detect(debug = args['debug']>0)
    ocr_read = OCR(debug=args["debug"]>0)
    seg_read = character_segmentation(n_clusters=3, debug=args["debug"]>0)
    img_path = sorted(list(paths.list_images(args["input"])))[:100]
    for _, imagePath in enumerate(img_path, start=1):
        # load the input image from disk and resize it
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        # apply automatic license plate recognition
        lpCnt, ROI_list = lpd.crop(image)
        print('----------------------------------------------')
        print(f'Hinh thu: {_}')
        if len(lpCnt) == 0:
            print("Khong tim thay bien so")
            cv2.putText(image, "Khong tim thay bien so", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f"Hinh {_}", image)
            cv2.waitKey(stop)
            cv2.destroyAllWindows()
            continue
        flag = False
        if args["segment"] == 0:
            for (i, r) in enumerate(ROI_list):
                    char = ocr_read.read(r)
                    if len(char) == 9 or len(char) == 10:
                        count_len +=1;
                    if len(char) > 2:
                        cnt += 1
                        x, y, w, h = cv2.boundingRect(lpCnt[i])
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(image, '{}'.format(char.replace('\n', '-')), (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        cv2.imshow(f"Hinh {_}", image)
                        print('Bien so:\n' + char)
                        flag = True
            cv2.waitKey(stop)
            
        if args["segment"] == 1:
            for (i, r) in enumerate(ROI_list):
                    char = seg_read.segmentation(r)
                    if len(char) > 2:
                        cnt += 1
                        x, y, w, h = cv2.boundingRect(lpCnt[i])
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        cv2.putText(image, '{}'.format(char.replace('\n', '-')), (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        cv2.imshow(f"Hinh {_}", image)
                        print('Bien so:\n' + char)
                        flag = True
            cv2.waitKey(stop)

        if flag is False:
                print('Khong doc duoc bien so')
                cv2.putText(image, "Khong doc duoc bien so", (220, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(f"Output_{_}", image)
                cv2.waitKey(stop)
        cv2.destroyAllWindows()

        cv2.waitKey(stop)




    print('----------------------------------------------')
    print('Tong nhan dien duoc {} hinh'.format(cnt))
    print('Tong nhan dien duoc {} hinh (8~9)'.format(count_len))
    print('So luong hinh: {}'.format(len(img_path)))
    #print('Ti le nhan dien duoc (8~9): {:.2f}'.format(count_len / cnt))

if __name__ == "__main__":
    args = parse_args()
    main(args)
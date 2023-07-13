import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)


class OCR:
    def __init__(self, debug=False):
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        # check to see if we are in debug mode, and if so, show the
        # image with the supplied title
        if self.debug:
            cv2.imshow(title, image)
            # check to see if we should wait for a keypress
            if waitKey:
                cv2.waitKey(0)

    def read(self, img):
        # read text from license plate image without segmentation
        alp = reader.readtext(img, detail=0, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ.-')
        for i in range(len(alp)):
            try:
                alp[i] = alp[i].replace('.', '')
                alp[i] = alp[i].replace('-', '')
            except:
                continue

        return '\n'.join(alp)

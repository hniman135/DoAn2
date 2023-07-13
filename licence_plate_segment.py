from sklearn.cluster import KMeans
import numpy as np
import cv2
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

MIN_RATIO_AREA = 0.015
MAX_RATIO_AREA = 0.07
MIN_RATIO = 2
MAX_RATIO = 6


def KMeans_(img, n_clusters=3):
    """
    The function performs K-means clustering on an input image with a specified number of clusters and
    returns the resulting image.
    
    :param img: The input image that needs to be segmented using KMeans clustering
    :param n_clusters: The number of clusters to form in the KMeans algorithm. It determines how many
    distinct colors the resulting image will have, defaults to 3 (optional)
    :return: an image with the same shape as the input image, but with the pixel values replaced by the
    cluster center values obtained from KMeans clustering. The data type of the returned image is uint8.
    """
    nrow, ncol = img.shape
    g = img.reshape(nrow * ncol, -1)
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(g)
    t = k_means.cluster_centers_[k_means.labels_]
    img_res = t.reshape(nrow, ncol)
    img_res = img_res.astype(np.uint8)
    return img_res


class character_segmentation:
    def __init__(self, n_clusters=3, debug=False):
        self.n_clusters = n_clusters
        self.debug = debug

    def debug_imshow(self, title, image, waitKey=False):
        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)
    def segmentation(self, img):
        """
        This function performs image segmentation using KMeans clustering and Canny edge detection, and
        then finds the contours of the segmented regions.
        
        :param img: The input image that needs to be segmented
        """
        self.debug_imshow("First", img)
        height, width = img.shape
        seg_img = KMeans_(img, self.n_clusters)
        area = seg_img.shape[0] * seg_img.shape[1]
        seg_img = seg_img.astype(np.uint8)
        self.debug_imshow("Kmean", seg_img)
        ret, thresh = cv2.threshold(seg_img, 90, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh, (5, 5), 0)
        im_bw = cv2.Canny(blur, 10, 200)
        self.debug_imshow('Canny', im_bw)

        img_BGR = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        if self.debug:
            cnts = cv2.drawContours(img_BGR.copy(), contours, -1, (0, 0, 255), 3)
            self.debug_imshow('Contour', cnts)

        # `new_contours` is a list that stores the contours of the segmented regions that meet certain
        # criteria. The function loops through all the contours obtained from the Canny edge detection
        # and KMeans clustering, and for each contour, it calculates its bounding rectangle using
        # `cv2.boundingRect(c)`. If the area of the bounding rectangle falls within a certain range
        # (`MIN_RATIO_AREA * area <= w * h <= MAX_RATIO_AREA * area`) and the height-to-width ratio of
        # the bounding rectangle also falls within a certain range (`MIN_RATIO <= h / w <=
        # MAX_RATIO`), then the contour is considered a valid character region and is added to
        # `new_contours`.
        new_contours = []

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if MIN_RATIO_AREA * area <= w * h <= MAX_RATIO_AREA * area \
                    and MIN_RATIO <= h / w <= MAX_RATIO:
                new_contours.append(c)

        # `chars` is a list that stores the segmented characters obtained from the input image. The
        # function loops through all the contours obtained from the Canny edge detection and KMeans
        # clustering, and for each contour, it calculates its bounding rectangle using
        # `cv2.boundingRect(c)`. It then applies padding to the bounding rectangle to help with OCR
        # accuracy. If the padding causes the bounding rectangle to extend beyond the boundaries of
        # the input image, the function skips that contour. Otherwise, it extracts the character region 
        # from the input image using the padded bounding rectangle and appends it to `chars`
        # The function also stores the x and y coordinates of the top-left corner of each bounding
        # rectangle in `X` and `Y`, respectively.
        chars = []
        X, Y = [], []
        # segment image into characters
        for c in new_contours:
            (x, y, w, h) = cv2.boundingRect(c)

            # padding to help with OCR accuracy
            x_cut = x - 5 if x - 5 > 0 else 0
            y_cut = y - 5 if y - 5 > 0 else 0
            w_cut = w + 5 if x + w + 10 < width else w
            h_cut = h + 5 if y + h + 10 < height else h

            if x_cut == 0 or x + w_cut == width:
                continue
            chars.append(img[y_cut:y + h_cut, x_cut:x + w_cut])
            X.append(x)
            Y.append(y)

        # This code segment is dividing the segmented characters into two groups, `chars_top` and
        # `chars_bottom`, based on their position in the image. If the y-coordinate of a character's
        # bounding rectangle is less than half the height of the image minus 30, it is considered to
        # be in the top group (`chars_top`), otherwise it is in the bottom group (`chars_bottom`). The
        # code then sorts both groups of characters based on their x-coordinate (`X[i]`) in ascending
        # order using the `sorted()` function with a lambda function as the key. The resulting sorted
        # lists are stored in `chars_top` and `chars_bottom`, respectively. `s_top` and `s_bottom` are
        # empty lists that will be used to store the OCR results for the characters in `chars_top` and
        # `chars_bottom`, respectively.
        s_top, s_bottom = [], []
        chars_top = []
        chars_bottom = []

        for i, c in enumerate(chars):
            if Y[i] < height / 2 - 30:
                chars_top.append([c, X[i], Y[i]])
            else:
                chars_bottom.append([c, X[i], Y[i]])

        chars_top = sorted(chars_top, key=lambda x: x[1])
        chars_bottom = sorted(chars_bottom, key=lambda x: x[1])

        if len(chars_top) < 4:
            for i in range(len(chars_top)):
                h, w = chars_top[i][0].shape[:2]
                s = reader.recognize(chars_top[i][0], detail=0, allowlist='0123456789ABCDEFGHKLMNPRSTUVXYZ')
                if s is not None and self.debug:
                    cv2.putText(img_BGR, *s, (chars_top[i][1] + 8, chars_top[i][2] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)
                    cv2.rectangle(img_BGR, (chars_top[i][1], chars_top[i][2]),
                                  (chars_top[i][1] + w - 5, chars_top[i][2] + h - 5), (0, 255, 0), 2)

                    self.debug_imshow('char', chars_top[i][0], waitKey=True)

                s_top.append(*s)
        else:
            for i in range(len(chars_top)):
                h, w = chars_top[i][0].shape[:2]
                if i == 2:
                    s = reader.recognize(chars_top[i][0], detail=0, allowlist='ABCDEFGHKLMNPRSTUVXYZ')
                else:
                    s = reader.recognize(chars_top[i][0], detail=0, allowlist='0123456789')
                if s is not None and self.debug:
                    cv2.putText(img_BGR, *s, (chars_top[i][1] + 8, chars_top[i][2] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2)
                    cv2.rectangle(img_BGR, (chars_top[i][1], chars_top[i][2]),
                                  (chars_top[i][1] + w - 5, chars_top[i][2] + h - 5), (0, 255, 0), 2)

                    self.debug_imshow('char', chars_top[i][0], waitKey=True)

                s_top.append(*s)

        for i in range(len(chars_bottom)):
            h, w = chars_bottom[i][0].shape[:2]
            s = reader.recognize(chars_bottom[i][0], detail=0, allowlist='0123456789')
            if s is not None and self.debug:
                cv2.putText(img_BGR, *s, (chars_bottom[i][1] + 8, chars_bottom[i][2] + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)
                cv2.rectangle(img_BGR, (chars_bottom[i][1], chars_bottom[i][2]),
                              (chars_bottom[i][1] + w - 5, chars_bottom[i][2] + h - 5), (0, 255, 0), 2)

                self.debug_imshow('char', chars_bottom[i][0], waitKey=True)

            s_bottom.append(*s)
        return ''.join(s_top) + ''.join(s_bottom)
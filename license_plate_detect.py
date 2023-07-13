import numpy as np
import imutils
import cv2

class license_plate_detect:
    def __init__(self, minAR=1.1, maxAR=1.6, minContourArea=1000, debug=False):
        """
        This is the initialization function for a class that sets various parameters such as minimum
        aspect ratio, maximum aspect ratio, minimum contour area, and debug mode.
        
        :param minAR: The minimum aspect ratio of a contour that will be considered a possible license
        plate. The aspect ratio is the ratio of the width to the height of the contour's bounding box
        :param maxAR: The maximum aspect ratio allowed for a contour to be considered a valid target.
        Aspect ratio is the ratio of the width to the height of the contour's bounding box
        :param minContourArea: The minimum area of a contour (in pixels) that will be considered as a
        potential license plate region. Any contour with an area smaller than this value will be
        ignored, defaults to 1000 (optional)
        :param debug: A boolean parameter that determines whether or not to display debug information
        during the execution of the code. If set to True, the code will display additional information
        that can be useful for debugging purposes, defaults to False (optional)
        """
        self.minAR = minAR
        self.maxAR = maxAR
        self.minContourArea = minContourArea
        self.debug = debug
		
    def debug_imshow(self, title, image, waitKey=False):
        """
        This function displays an image with a given title if the program is in debug mode and waits for
        a keypress if specified.
        
        :param title: A string that will be used as the title of the window in which the image will be
        displayed
        :param image: The image to be displayed in the OpenCV window
        :param waitKey: A boolean parameter that determines whether the program should wait for a
        keypress after showing the image. If set to True, the program will wait for a keypress before
        continuing. If set to False, the program will not wait and will continue immediately, defaults
        to False (optional)
        """

        if self.debug:
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        """
        This function locates license plate candidates in an image using various morphological
        operations and contour detection.
        
        :param gray: a grayscale image of the input image
        :param keep: The number of contours to keep after sorting them by size in descending order,
        defaults to 5 (optional)
        :return: a tuple containing the gray image and a list of contours (cnts).
        """

        # This code is performing morphological operations on the grayscale image `gray` to enhance the
        # contrast of the image and highlight potential license plate regions.
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        gray_contr = cv2.subtract(gray, blackhat)
        self.debug_imshow("Blackhat", blackhat)


        # This code is performing morphological operations on the grayscale image `gray` to enhance
        # the contrast of the image and highlight potential license plate regions.
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # This code is performing edge detection on the image `blackhat` using the Sobel operator. The
        # resulting gradient image is then normalized to have pixel values between 0 and 255, and is
        # converted to an 8-bit unsigned integer format. The resulting image is displayed in a window
        # titled "Scharr" if the program is in debug mode.
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)


        # This code is performing image processing operations on the gradient image `gradX`. First, a
        # Gaussian blur is applied to the image with a kernel size of (7, 7) and a sigma value of 0.
        # Then, a morphological closing operation is performed on the blurred image using a
        # rectangular kernel `rectKern`. Finally, a binary thresholding operation is applied to the
        # result using Otsu's method, and the resulting binary image is stored in the variable
        # `thresh`. The resulting image is displayed in a window titled "Grad Thresh" if the program
        # is in debug mode.
        gradX = cv2.GaussianBlur(gradX, (7, 7), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", gradX)


        # This code is performing erosion and dilation operations on the binary thresholded image
        # `thresh`. The `cv2.erode()` function is used to erode away the boundaries of the foreground
        # objects in the image, while the `cv2.dilate()` function is used to dilate the boundaries of
        # the foreground objects. These operations are performed to remove noise and fill in gaps in
        # the license plate candidates. The `iterations` parameter specifies the number of times the
        # operation should be applied. The resulting image is displayed in a window titled "Grad
        # Erode/Dilate" if the program is in debug mode.
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=3)
        self.debug_imshow("Grad Erode/Dilate", thresh)


        # This code is performing additional image processing operations on the binary thresholded
        # image `thresh`. First, it applies a bitwise AND operation between `thresh` and `light` to
        # remove any areas that are not illuminated. Then, it performs a dilation operation on the
        # resulting image to fill in any gaps in the license plate candidates. Finally, it performs an
        # erosion operation on the image to remove any small noise regions. The resulting image is
        # displayed in a window titled "Final" if the program is in debug mode.
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=3)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, waitKey=True)


        # This code is finding contours in the binary thresholded image `thresh` using the
        # `cv2.findContours()` function. The `cv2.RETR_EXTERNAL` flag retrieves only the external
        # contours, while the `cv2.CHAIN_APPROX_SIMPLE` flag compresses horizontal, vertical, and
        # diagonal segments and leaves only their end points. The resulting contours are then processed
        # using the `imutils.grab_contours()` function to ensure compatibility with both OpenCV 2 and
        # OpenCV 3. Finally, the contours are sorted in descending order by area using the `sorted()`
        # function and only the top `keep` contours are kept.
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        return gray_contr, cnts
    
    def locate_license_plate(self, gray, candidates):
        """
        This function takes in a grayscale image and a list of contour candidates, filters out the
        candidates that do not meet certain size and aspect ratio requirements, and returns a list of
        remaining contour candidates that are likely to be license plates.
        
        :param gray: a grayscale image of the license plate region
        :param candidates: a list of contours that are potential license plate candidates in an image
        :return: a list of contours that meet certain criteria (size and aspect ratio) from a list of
        candidate contours.
        """
        lpCnt = []

        # This code is displaying the potential license plate candidates on the input image if the
        # program is in debug mode. It converts the grayscale image `gray` to a BGR image `img_BGR`
        # using the `cv2.cvtColor()` function, and then draws the contours of the potential license
        # plate candidates on the image using the `cv2.drawContours()` function. The resulting image
        # is then displayed in a window titled "Candidates" using the `self.debug_imshow()` function.
        if self.debug:
            img_BGR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cnts = cv2.drawContours(img_BGR.copy(), candidates, -1, (0, 255, 0), 2)
            self.debug_imshow("Candidates", cnts)

        # This code is iterating through a list of contour candidates and calculating the aspect ratio
        # of each contour using its bounding rectangle dimensions. It then checks if the aspect ratio
        # falls within a certain range (`self.minAR` to `self.maxAR`) and if the area of the contour
        # is greater than a certain threshold (`self.minContourArea`). If the contour meets these
        # criteria, it is added to a list of remaining contour candidates (`lpCnt`).
        for c in candidates:
            # tính toán kích thước của contour
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if self.minAR <= ar <= self.maxAR and w * h > self.minContourArea:
                lpCnt.append(c)

        return lpCnt
    
    def crop(self, image):
        """
        This function crops the license plate region from an input image and returns the cropped region
        as well as the coordinates of the license plate contour.
        
        :param image: The input image from which license plate needs to be cropped
        :return: two lists: lpCnt and ROI_list. lpCnt contains the contours of the located license
        plate(s) in the input image, while ROI_list contains the cropped and resized license plate
        image(s).
        """
        # The above code is performing the following operations:
        # 1. Converting an input image from BGR color space to grayscale using OpenCV's cv2.cvtColor()
        # function.
        # 2. Locating potential license plate candidates in the grayscale image using a custom
        # function called locate_license_plate_candidates().
        # 3. Locating the actual license plate contour in the grayscale image using a custom function
        # called locate_license_plate().
        # 4. If no license plate contour is found, the function returns empty lists for both the
        # license plate contour and the license plate candidates.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray, candidates = self.locate_license_plate_candidates(gray)
        lpCnt = self.locate_license_plate(gray, candidates)
        if len(lpCnt) == 0:
            return [], []

        ROI_list = []
        # The above code is iterating through a list of contours stored in the variable `lpCnt`. For
        # each contour, it is finding the minimum area rectangle that can enclose the contour using
        # the `cv2.minAreaRect()` function. Then, it is finding the four corner points of the
        # rectangle using the `cv2.boxPoints()` function and converting them to integers using
        # `np.int0()`. The resulting points can be used to draw the rectangle around the contour.
        for i in lpCnt:
            rect = cv2.minAreaRect(i)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # The above code is calculating the center and size of a rotated rectangle (specified by
            # the variable "rect") that encloses a set of points (specified by the variable "box"). It
            # first extracts the width and height of the rectangle from the "rect" variable. Then, it
            # finds the minimum and maximum x and y coordinates of the points in "box" to determine
            # the bounding box of the points. It also adjusts the angle of the rectangle if it is
            # greater than 45 degrees. Finally, it calculates the center and size of the rectangle
            # using the bounding box coordinates.
            W = rect[1][0]
            H = rect[1][1]

            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            angle = rect[2]
            if angle > 45:
                angle -= 90

            # The above code is performing image processing operations using OpenCV library in Python.
            # It is taking an input image and performing the following operations:
            # 1. Finding the center and size of a rectangle defined by two points (x1, y1) and (x2,
            # y2).
            # 2. Applying rotation to the rectangle by a given angle.
            # 3. Cropping the rotated rectangle from the input image.
            # 4. Resizing the cropped image to a fixed width of 200 pixels.
            # 5. Adding the resized image to a list of ROI (Region of Interest) images.
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
            cropped = cv2.getRectSubPix(gray, size, center)
            cropped = cv2.warpAffine(cropped, M, size)
            croppedW = H if H > W else W
            croppedH = H if H < W else W
            ROI = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))
            ROI = imutils.resize(ROI, width=200)
            ROI_list.append(ROI)
        return lpCnt, ROI_list
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
from scipy.interpolate import splprep, splev
from classify import *

save_path = "./cropped/"
#Path of video file to be read
video_read_path='speed1.mp4'

#Path of video file to be written
video_write_path='speed1.avi'

#Window Name
window_name='Input Video'

#Escape ASCII Keycode
esc_keycode=27

#Create an object of VideoCapture class to read video file
video_read = cv2.VideoCapture(video_read_path)
    # Check if video file is loaded successfully
if (video_read.isOpened()== True):
    #Frames per second in videofile. get method in VideoCapture class.
    fps = video_read.get(cv2.CAP_PROP_FPS)
    #Width and height of frames in video file
    size = (int(video_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #Create an object of VideoWriter class to write video file.
    #cv2.CV_FOURCC('I','4','2','0') = uncompressed YUV, 4:2:0 chroma subsampled. (.avi)
    #cv2.CV_FOURCC('P','I','M','1') = MPEG-1(.avi)
    #cv2.CV_FOURCC('M','J','P','G') = motion-JPEG(.avi)
    #cv2.CV_FOURCC('T','H','E','O') = Ogg-Vorbis(.ogv)
    #cv2.CV_FOURCC('F','L','V','1') = Flash video (.flv)
    #cv2.CV_FOURCC('M','P','4','V') = MPEG encoding (.mp4)
    #Also this form is too valid cv2.VideoWriter_fourcc(*'MJPG')
    #video_write = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    video_write = cv2.VideoWriter(video_write_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    #Set display frame rate
    display_rate = (int) (1/fps * 1000)
    #Create a Window
    #cv2.WINDOW_NORMAL = Enables window to resize.
    #cv2.WINDOW_AUTOSIZE = Default flag. Auto resizes window size to fit an image.
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    counter = 0
    #Read first frame from video. Return Boolean value if it succesfully reads the frame in state and captured frame in cap_frame
    state, img = video_read.read()
    #Loop untill all frames from video file are read
    imageC = 1
    xp,yp=0,0
    while state:

        #My Code

        counter+=1
        output = img.copy()
        output2 = img.copy()
        output3 = img.copy()
        # img = cv2.GaussianBlur(frame,(5,5),0)

        # MSER
        """""
        mser = cv2.MSER_create()
        regions = mser.detectRegions(img)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))
        cv2.imshow('img', vis)
        """

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv', hsv)

        # range
        lower_red_1 = np.array([0, 50, 50])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 50, 50])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        mask = mask1 + mask2

        red = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow('res1', red)

        gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('res2', gray)

        # blur = cv2.GaussianBlur(gray,(3,3),0)
        # cv2.imshow('res3', blur)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)

        close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('res3', close)
        median = cv2.medianBlur(close, 3)

        # cv2.imshow("i",np.hstack([gray,median,close]))
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # sharp = cv2.filter2D(median, -1, kernel)
        # cv2.imshow('res5', sharp)

        im2, contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if ((len(approx) > 8) & (area > 130)):
                contour_list.append(contour)

        # smoothening of contours
        '''
        smoothened = []
        for contour in contour_list:
            x, y = contour.T
            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1)
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), 25)
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)
            # Convert it back to numpy format for opencv to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))


        cv2.drawContours(output, smoothened, -1, (0, 255, 0), 3)
        cv2.fillPoly(output, pts=smoothened, color=(0, 255, 0))
        '''
        # Getting circles

        threshold = 0.50
        circle_list = []
        for contour in contour_list:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            metric = 4 * 3.14 * area / pow(perimeter, 2)
            if (metric > threshold):
                circle_list.append(contour)
        #print(circle_list)
        # cv2.drawContours(output2, circle_list, -1, (0, 255, 0), 3)
        # cv2.fillPoly(output2, pts=circle_list, color=(0, 255, 0))

        # Drawing Rectangle
        '''
        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        height, width, _ = img.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        # computes the bounding box for the contour, and draws it on the frame,
        for contour, hier in zip(circle_list, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            if w > 80 and h > 80:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if max_x - min_x > 0 and max_y - min_y > 0:
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
        '''
        xm, ym, wm, hm, = 0, 0, 0, 0,
        for contour in circle_list:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)
            #print("m = ", [xm, ym, wm, hm])
            #print([x, y, w, h])
            if(xp==x and yp==y):
                continue
            elif (xm == 0):
                xm, ym, wm, hm = x, y, w, h
            else:
                if ((abs(xm - x) < 30) and (abs(ym - y) < 30) and (abs(wm - w)>3) and (abs(hm -h)>3)):
                    # xmean, ymean = (xm+x)//2,(ym+y)//2
                    # wmax, hmax = max(wm,w),max(hm,h)
                    if (wm > w):
                        crop = img[(ym):(ym + hm), (xm):(xm + wm)].copy()
                        xp,yp = xm,ym
                    else:
                        crop = img[(y):(y + h), (x):(x + w)].copy()
                        xp, yp = x, y

                    crop = cv2.resize(crop, (32,32))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    preprocessed = preprocessed(crop)
                    print(preprocessed.shape)
                    preprocessed = preprocessed.reshape(32,32,1)
                    y_prob, y_pred = y_predict_model(preprocessed)
                    print(y_prob+"  "+y_pred)


                    #cv2.imwrite(os.path.join(save_path, str(imageC) + ".jpg"), crop)
                    print("IMAGE SAVED FOR CLASSIFICATION ", str(imageC))
                    imageC += 1
            xm, ym, wm, hm = x, y, w, h
            # draw rectangle around contour on original image
            cv2.rectangle(output3, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Write frame
        # if((length/counter)%100 == 0):
        print(counter)

        # Display frame
        cv2.imshow(window_name,output3)
        #Write method from VideoWriter. This writes frame to video file
        video_write.write(output3)
        #Read next frame from video
        state, img = video_read.read()
        #Check if any key is pressed.
        k = cv2.waitKey(display_rate)
        #Check if ESC key is pressed. ASCII Keycode of ESC=27
        if k == esc_keycode:
            #Destroy Window
            cv2.destroyWindow(window_name)
            break
    #Closes Video file
    video_read.release()
    video_write.release()
else:
    print("Error opening video stream or file")

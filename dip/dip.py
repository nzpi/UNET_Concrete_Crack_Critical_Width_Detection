from nn import nn
from skimage.morphology import skeletonize
import setting.constant as const
import dip.image as im
import importlib
import cv2
import numpy as np


def improcessor(image, label=None):
    pp = importlib.import_module("%s.%s.%s" % (const.dn_DIP, const.dn_PROCESSING, const.IMG_PROCESSING))
    image = pp.image_preprocessor(image)

    if (label is not None):
        label = pp.label_preprocessor(label)
    
    return (image, label)

def preprocessor(image, label=None):
    pp = importlib.import_module("%s.%s.%s" % (const.dn_DIP, const.dn_PROCESSING, const.IMG_PROCESSING))
    image = cv2.resize(image, dsize=const.IMAGE_SIZE[:2])
    image = pp.image_preprocessor(image)

    if (label is not None):
        label = cv2.resize(label, dsize=const.IMAGE_SIZE[:2])
        label = pp.label_preprocessor(label)
    
    return (image, label)


def posprocessor(original, image):
    pp = importlib.import_module("%s.%s.%s" % (const.dn_DIP, const.dn_PROCESSING, const.IMG_PROCESSING))
    image = cv2.resize(image, original.shape[:2][::-1])
    image = pp.posprocessor(image)
    return im.threshold(image)

def overlay(image, layer):
    return im.overlay(image, layer)

def measure(image, label=None):
    pp = importlib.import_module("%s.%s.%s" % (const.dn_DIP, const.dn_PROCESSING, const.IMG_PROCESSING))  
    imgray = np.invert(image)
    image = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
    original = imgray

    ## Watershed contour ## 
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(imgray,cv2.MORPH_OPEN,kernel, iterations = 3)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(image,markers)
    image[markers == -1] = [0,0,255]
    
    markers1 = markers.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    c, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    

    for i in c:
        area = cv2.contourArea(i)
        if area <150: 
            original = cv2.fillPoly(original, pts =[i], color=(0,0,0))
            image = cv2.fillPoly(image, pts =[i], color=(0,0,0))
        else:
            image = cv2.drawContours(image, [i], -1, (0,0,0), 2)

    #original contour
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    closing = cv2.morphologyEx(imgray,cv2.MORPH_CLOSE,kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(imgray, 1, 2) 

    ret, thresh = cv2.threshold(image, 240, 255, cv2.THRESH_BINARY)
    image[thresh ==255] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    i = 0
    box = None

    for c in contours:
        if cv2.contourArea(c) > 150:
            contour = cv2.fillPoly(image, pts =[c], color=(255,255,255))
            #determine the properties of contour area
            rect = cv2.minAreaRect(c)
            area = cv2.contourArea(c)

            # Skeletonisation approach
            skeleton = im.skeleton(contour)
            points = np.where(skeleton == 255)
            length = cv2.countNonZero(skeleton)
            width = area/length
            contour[thresh ==255] = 0
            # Coordinate approach
            # max length of contour
            #left_max = tuple(c[c[:, :, 0].argmin()][0])
            #right_max = tuple(c[c[:, :, 0].argmax()][0])
            #l = np.sqrt( (right_max[0]-left_max[0])**2 + ((right_max[1] - left_max[1])**2) )

            # max width of contour
            #top_max = tuple(c[c[:, :, 1].argmin()][0])
            #bottom_max = tuple(c[c[:, :, 1].argmax()][0])
            #w = np.sqrt( (top_max[0]-bottom_max[0])**2 + (top_max[1] - bottom_max[1])**2)

            #width = area/max(l,w)
            # check largest width of image

            if width > i:
                a = area
                i = width
                length = length
                contour_critical = c
                contour_length = points
                # drawing box around areas segmented
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # coordinates for length
                width_critical = width
                length_critical = length
                top_max_critical = top_max
                bottom_max_critical = bottom_max
                left_max_critical = left_max
                right_max_critical = right_max

    closing = cv2.morphologyEx(original,cv2.MORPH_CLOSE,kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(closing, 1, 2) 

    a = 0
    w = None
    l = None

    for c in contours:
        if cv2.contourArea(c) > 100:
            image = cv2.fillPoly(image, pts =[c], color=(255,255,255))
            image = cv2.drawContours(image, [c], -1, (0,255,0), 1)
            # finding width of contour based on length and area

            # Coordinate approach 
            # max length of contour
            left_max = tuple(c[c[:, :, 0].argmin()][0])
            right_max = tuple(c[c[:, :, 0].argmax()][0])
            l = np.sqrt( (right_max[0]-left_max[0])**2 + ((right_max[1] - left_max[1])**2) )

            # max width of contour
            top_max = tuple(c[c[:, :, 1].argmin()][0])
            bottom_max = tuple(c[c[:, :, 1].argmax()][0])
            w = np.sqrt( (top_max[0]-bottom_max[0])**2 + (top_max[1] - bottom_max[1])**2)
            width = area/max(l,w)

            if cv2.contourArea(c) > a:
                a = cv2.contourArea(c)
                top_max = top_max
                bottom_max = bottom_max
                left_max = left_max
                right_max = right_max
                l = l
                w = w

    #Entire crack length segment
    #if w is not None:
       # if w > l:
            #l = w
            #top to bottom
            #image = cv2.circle(image, top_max, 5, (0, 255, 0), -1) 
            #image = cv2.circle(image, bottom_max, 5, (0, 255, 0), -1)  
            #image = cv2.line(image, top_max, bottom_max, (0,255,255), 2)
            #image = cv2.putText(image, 'Max Length: ' + str(l), (10,70), font, 0.3, (0,255,255),1 ,cv2.LINE_AA)

        #else:
            # left to right
            #image = cv2.circle(image, left_max, 5, (0, 255, 0), -1) 
            #image = cv2.circle(image, right_max, 5, (0, 255, 0), -1) 
            #image = cv2.line(image, left_max, right_max, (0,255,255), 2)
            #image = cv2.putText(image, 'Max Length: ' + str(l), (10,70), font, 0.3, (0,255,255),1 ,cv2.LINE_AA)

    
    if box is not None:
        image = cv2.drawContours(image, [contour_critical], -1, (0,0,255), 2)
        image[contour_length] = [0,0,255]
        image = cv2.drawContours(image,[box],0,(0,255,0),2)
        #image = cv2.putText(image, 'Length: ' + str(length), (10,10), font, 0.3, (0,0,255),1 ,cv2.LINE_AA)
        image = cv2.putText(image, 'Width: ' + str(i), (10,30), font, 0.3, (255,255,255),1 ,cv2.LINE_AA)
        image = cv2.putText(image, 'Area: '+ str(a), (10,10), font, 0.3, (255,255,255),1 ,cv2.LINE_AA)
        #if i > 
        #image = cv2.putText(image, 'Width: ' + str(i), (10,30), font, 0.3, (255,255,255),1 ,cv2.LINE_AA)


        #if width_critical > length_critical:
            #length_critical = width_critical
            #top to bottom
            #image = cv2.circle(image, top_max_critical, 5, (0, 0, 255), -1) 
            #image = cv2.circle(image, bottom_max_critical, 5, (0, 0, 255), -1)  
            #image = cv2.line(image, top_max_critical, bottom_max_critical, (255,255,0), 2)

        #else:
            # left to right
            #image = cv2.circle(image, left_max_critical, 5, (0, 0, 255), -1) 
            #image = cv2.circle(image, right_max_critical, 5, (0, 0, 255), -1) 
            #image = cv2.line(image, left_max_critical, right_max_critical, (255,255,0), 2)

    if (label is not None):
        label = cv2.resize(label, dsize=const.IMAGE_SIZE[:2])
        label = pp.label_preprocessor(label)
    
    return (image, label, i)
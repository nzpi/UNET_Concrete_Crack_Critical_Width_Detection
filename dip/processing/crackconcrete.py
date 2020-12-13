import dip.image as im
import numpy as np
import cv2

def image_preprocessor(image):
    image = cv2.bitwise_not(image)
    
    image = im.gauss_filter(image, (3,3))
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    image = im.equalize_light(image)

    image = im.light(image, bright=-20, contrast=-20)
    #image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = im.gauss_filter(image, (3,3))

    image = im.threshold(image, min_limit=127)
    #kernel = np.ones((4,4),np.uint8)
    #closing = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel, iterations = 3)
    image = im.otsu(image)
    
    return image

def label_preprocessor(label):
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    label = im.threshold(label, min_limit=127)
    return label

def posprocessor(image):
    return im.threshold(image)
from cv2 import cv2 as cv
import numpy as np
import os
import sys
from matplotlib import pyplot as plt


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
qwidth = width/4
qheight = height/4
init = True

print("width:",width,"height:",height,"qw:",qwidth,"qh:",qheight)

while True:
    ret, frame = cap.read()
    if not ret:
        print("cant recieve frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #edges = cv.Canny(frame, 100, 200)

    if init:
        #template = gray[int(qwidth):int(qwidth*3),int(qheight):int(qheight*3)]
        template = frame[int(qwidth):int(qwidth*3),int(qheight):int(qheight*3)]
        init = False
        w, h = template.shape[0:2]

    
    res = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
    #template = gray[int(qwidth):int(qwidth*3),int(qheight):int(qheight*3)]
    template = frame[int(qwidth):int(qwidth*3),int(qheight):int(qheight*3)]
    cv.rectangle(gray, max_loc, (max_loc[0]+w, max_loc[1]+h), 255,2)
    #cv.circle(gray, (qwidth,qheight), qheight, 255, -1)

    if max_loc[1] < qwidth:
        if max_loc[0] < qheight:
            cv.circle(gray, (5,5), 5, 255, -1)
        else:
            cv.circle(gray, (width-5,5), 5, 255, -1)
    else:
        if max_loc[0] < qheight:
            cv.circle(gray, (5,height-5), 5, 255, -1)
        else:
            cv.circle(gray, (width-5,height-5), 5, 255, -1)
    xshift, yshift = qwidth-max_loc[1], qheight-max_loc[0]
    cv.putText(gray,str(xshift)+","+str(yshift),(300,30),cv.FONT_HERSHEY_SIMPLEX,1,255)
    
    
    #cv.rectangle(gray, (qwidth,qheight), (qwidth*3,qheight*3), (255,0,0),3)
    cv.imshow('raw', frame)
    cv.imshow('gray', gray)
    cv.imshow('res', res)
    #cv.imshow('edges', edges)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
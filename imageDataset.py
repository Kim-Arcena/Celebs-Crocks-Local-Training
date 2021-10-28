import re
import numpy as np
import cv2
import json
import time
import os

path = "cropped"
count = 1
faces_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')                    
cap = cv2.VideoCapture('SYND 18 1 77 EXCLUSIVE INTERVIEW WITH PRESIDENT MARCOS OF PHILIPPINES IN MANILA.mp4')           #video to be converted         

while(cap.isOpened()):
    ret, frame = cap.read()
    count = count+ 1
    if(count % 5 == 0):             #for every 5 frames in the vid
        print(count)
        if(ret ==True):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faces_cascade.detectMultiScale(gray, 1.3,4)
            for (x,y,w,h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                cv2.imwrite(path+"/"+str(count)+".jpg", roi_color)
        else:
            cap.release()        
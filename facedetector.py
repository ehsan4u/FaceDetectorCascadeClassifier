# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:47:03 2019
@author: Ehsan
"""

import cv2

def detect(gray, frame): 
    #testing
    faces_d = faceCascade_default.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1,1))
    for (x, y, w, h) in faces_d: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 255, 0), 2) 
    
    
    #nose not detecting
    nose = nose_cascade.detectMultiScale(gray)
    for (nx,ny,nw,nh) in nose:
           cv2.rectangle(gray,(nx,ny),(nx+nw,ny+nh),(105,120,0),2)
       
     
    faces = faceCascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
  
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        #eyes        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        #eye not working
        righteye = righteye_2splits.detectMultiScale(gray)
        for (nx,ny,nw,nh) in righteye:
           cv2.rectangle(gray,(nx,ny),(nx+nw,ny+nh),(255, 0, 0),2)
    
    return frame
new_path = 'C:/Anaconda3/Library/etc/haarcascades/'
faceCascade_default = cv2.CascadeClassifier(new_path+'haarcascade_frontalface_default.xml')

faceCascade = cv2.CascadeClassifier(new_path+'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(new_path+'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(new_path+'haarcascade_smile.xml')

nose_cascade = cv2.CascadeClassifier(new_path+'haarcascade_mcs_nose.xml')
 
righteye_2splits = cv2.CascadeClassifier(new_path+'haarcascade_righteye_2splits.xml')


video_capture = cv2.VideoCapture(0) 
while True: 
   # Captures video_capture frame by frame 
    _, frame = video_capture.read()  
  
    # To capture image in monochrome                     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
      
    # calls the detect() function     
    canvas = detect(gray, frame)    
  
    # Displays the result on camera feed                      
    cv2.imshow('Video', canvas)  
  
    # The control breaks once q key is pressed                         
    if cv2.waitKey(1) & 0xff == ord('q'):                
        break
  
# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 

 
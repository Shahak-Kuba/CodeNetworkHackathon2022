import cv2
import numpy as np

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray_picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#make picture gray
    faces = face_cascade.detectMultiScale(gray_picture, 1.1, 5)

    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),5)
        gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
        face = frame[y:y+h, x:x+w] # cut the face frame out
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 10)
        '''
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,0,255),7)'''
        #eye_frame = frame[eyes[0][0]:eyes[1][0]+eyes[1][2], eyes[0][1]:eyes[1][1]+eyes[1][3]]

    h,w = face.shape[:2]
    cut_face = face[round(h/5):round(3*h/5), 0:w]
    cv2.imshow("webcam", cut_face)

    c = cv2.waitKey(1)
    if c == 27:
        break

    
#cv2.imshow('my image',original_img)
cv2.destroyAllWindows()


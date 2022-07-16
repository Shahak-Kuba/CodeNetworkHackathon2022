import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# original image 
original_img = cv2.imread("Photos/IMG_2304.jpg", cv2.IMREAD_COLOR)
gray_picture = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)#make picture gray
faces = face_cascade.detectMultiScale(gray_picture, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(original_img,(x,y),(x+w,y+h),(255,255,0),5)
    
    gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
    face = original_img[y:y+h, x:x+w] # cut the face frame out
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 40)

    i = 0
    for (ex,ey,ew,eh) in eyes: 
        cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,0,255),20)
    




cv2.imshow('my image',original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


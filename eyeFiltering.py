import cv2
import numpy as np
import dlib

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Blob detection
detector = dlib.get_frontal_face_detector()

def error_print(i):
    face_error = "None"
    if(i == 0):
        face_error = "No face detected"
    if(i == 1):
        face_error = "Face detected"
    print(face_error)

while True:
    ret, frame = video.read()
    frame_width, frame_height = frame.shape[:2]
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray_picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#make picture gray
    faces = face_cascade.detectMultiScale(gray_picture, 1.1, 5)
    error = 0

    for (x,y,w,h) in faces:
        error = 1
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),5)
        gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
        face = frame[y:y+h, x:x+w] # cut the face frame out
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 10)
        '''
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,0,255),7)'''
        #eye_frame = frame[eyes[0][0]:eyes[1][0]+eyes[1][2], eyes[0][1]:eyes[1][1]+eyes[1][3]]

        h,w = face.shape[:2]
        cut_face = face[round(h/4):round(h/2), 0:w]
        eye_resized = cv2.resize(cut_face,(frame_width*3, frame_height//2))
        eye_resized = eye_resized[:, :eye_resized.shape[1]//2]
        
        rows, cols, _ = eye_resized.shape
        cut_face_grey = cv2.cvtColor(eye_resized,cv2.COLOR_BGR2GRAY)
        filtered_eyes = cv2.GaussianBlur(cut_face_grey, (3, 3), 0)
        _, threshold = cv2.threshold(filtered_eyes, 25, 255, cv2.THRESH_BINARY_INV)
        contours, _  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(eye_resized, [cnt], -1, (0, 0, 255), 3)
            cv2.line(eye_resized, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(eye_resized, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            break


        cv2.imshow("webcam", eye_resized)

        # blob detection 
        #cv2.imshow("webcam", eyes_with_keypoints)


    error_print(error)
    c = cv2.waitKey(1)
    if c == 27:
        break

    
#cv2.imshow('my image',original_img)
cv2.destroyAllWindows()


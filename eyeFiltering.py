import cv2
import numpy as np
import pyautogui as gui

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

eye_x_pos = 0
eye_y_pos = 0
prev_eye_x_pos = 0
prev_eye_y_pos = 0


prev_dir = "None"
curr_dir = "None"
new_dir = "None"

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

    if len(faces) == 0:
        continue

    for (x,y,w,h) in faces[np.argsort(faces[:, 2] * faces[:, 3])]:
        prev_eye_x_pos = eye_x_pos
        prev_eye_y_pos = eye_y_pos
        error = 1
        gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
        face = frame[y:y+h, x:x+w] # cut the face frame out

        h,w = face.shape[:2]
        cut_face = face[round(h/4):round(h/2), w//4:w//2]
        eye_resized = cv2.resize(cut_face,(frame_width*3, frame_height//2))
        eye_resized = eye_resized[:, :(3 * eye_resized.shape[1])//4]
        
        rows, cols, _ = eye_resized.shape
        cut_face_grey = cv2.cvtColor(eye_resized,cv2.COLOR_BGR2GRAY)
        filtered_eyes = cv2.GaussianBlur(cut_face_grey, (3, 3), 0)
        _, threshold = cv2.threshold(filtered_eyes, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(eye_resized, [cnt], -1, (0, 0, 255), 3)
            cv2.line(eye_resized, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(eye_resized, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            eye_x_pos = 0.6 * eye_x_pos + 0.4 * ( x + int(w/2) )
            eye_y_pos = 0.6 * eye_y_pos + 0.4 * ( y + int(h/2) )
            break


        if(eye_x_pos - prev_eye_x_pos > 50): # Left
            new_dir = "Left"
        elif(eye_x_pos - prev_eye_x_pos < -50): # Right
            new_dir = "Right"
        else:
            new_dir = "None"
        #if(eye_y_pos - prev_eye_y_pos > 0): # Down
        #    new_dir = "Down"
        #if(eye_y_pos - prev_eye_y_pos < 0): # Up
        #    new_dir = "Up"

        if(prev_dir == curr_dir):
            if(curr_dir == "Right"):
                gui.scroll(1)
            if(curr_dir == "Left"):
                gui.scroll(-1)
            if(curr_dir == "None"):
                pass

        prev_dir = curr_dir
        curr_dir = new_dir

        txt = "X: {0}, Y: {1}, prev direction: {2}, current direction: {3}".format(round(eye_x_pos),round(eye_y_pos),
        prev_dir, curr_dir)
        
        img_with_pos = cv2.putText(eye_resized, txt, ((frame_width//2) + 50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("EYE TRACKER", img_with_pos)



    error_print(error)
    c = cv2.waitKey(1)
    if c == 27:
        break

    
#cv2.imshow('my image',original_img)
cv2.destroyAllWindows()


from typing import Tuple
import cv2
from cv2 import COLOR_GRAY2BGR
import numpy as np
import pyautogui as gui
from collections import deque

SMOOTH_WINDOW = 20

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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def average_last_faces(last_faces: deque) -> Tuple[int, int, int, int]:
    if len(last_faces) < 5:
        return last_faces[0], [], []

    fs = np.array(last_faces)
    sizes = fs[:, 2] * fs[:, 3]

    smallest, biggest = min(sizes), max(sizes)

    fs = fs[np.argsort(sizes)]
    # if len(last_faces)>10:
    #     breakpoint()

    closest = np.abs(sizes-smallest) - np.abs(sizes-biggest)

    included = fs[closest > 0]
    excluded = fs[closest <= 0]


    x, y, w, h = included.mean(axis=0).round().astype(int)
    return (x, y, w, h), included, excluded

def error_print(i):
    face_error = "None"
    if(i == 0):
        face_error = "No face detected"
    if(i == 1):
        face_error = "Face detected"
    print(face_error)

last_faces = deque(maxlen=SMOOTH_WINDOW)
last_eye = deque(maxlen=SMOOTH_WINDOW // 2)

while True:
    ret, frame = video.read()
    frame_width, frame_height = frame.shape[:2]
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray_picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#make picture gray
    gray_picture = cv2.equalizeHist(gray_picture)
    faces = face_cascade.detectMultiScale(gray_picture, 1.1, 5)
    error = 0

    if len(faces) == 0:
        continue

    for (x,y,w,h) in faces[np.argsort(faces[:, 2] * faces[:, 3])][::-1][:1]:
        last_faces.append((x, y, w, h))
        (x, y, w, h), included, excluded = average_last_faces(last_faces)
        prev_eye_x_pos = eye_x_pos
        prev_eye_y_pos = eye_y_pos
        error = 1
        gray_face = gray_picture[y:y+h, x:x+w] # cut the gray face frame out
        face = frame[y:y+h, x:x+w] # cut the face frame out

        h,w = face.shape[:2]

        (fh1, fh2, fw1, fw2) = round(h/4), round(h/2), w//4, w//2
        cut_face = face[fh1:fh2, fw1:fw2]
        eye_resized = cv2.resize(cut_face,(frame_width*3, frame_height//2))
        eye_resized = eye_resized[:, :(3 * eye_resized.shape[1])//4]

        rows, cols, _ = eye_resized.shape
        cut_face_grey = cv2.cvtColor(eye_resized,cv2.COLOR_BGR2GRAY)
        filtered_eyes = cv2.GaussianBlur(cut_face_grey, (3, 3), 0)
        _, threshold = cv2.threshold(filtered_eyes, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        face_x, face_y = x, y

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            last_eye.append((x, y, w, h))
            (x, y, w, h), _, _ = average_last_faces(last_eye)
            cv2.drawContours(eye_resized, [cnt], -1, (0, 0, 255), 3)
            cv2.line(eye_resized, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(eye_resized, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            eye_x_pos = 0.6 * eye_x_pos + 0.4 * ( x + int(w/2) )
            eye_y_pos = 0.6 * eye_y_pos + 0.4 * ( y + int(h/2) )
            break

        _,ref_x = eye_resized.shape[:2]
        if(ref_x - eye_x_pos < 0 and eye_x_pos - prev_eye_x_pos > 50): # Left
            new_dir = "Left"
        elif(ref_x - eye_x_pos > 0 and eye_x_pos - prev_eye_x_pos < -50): # Right
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

        frame_copy = frame.copy()

        cv2.rectangle(frame_copy, (face_x + fw1, face_y+fh1), (face_x + fw2, face_y + fh2), (255, 0, 0), 2)

        if len(excluded) > 0:
            shades = np.linspace(80, 254, len(excluded))
            for shade, (x, y, w, h) in zip(shades, excluded):
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 0, int(shade)), 2)
        if len(included) > 0:
            shades = np.linspace(80, 254, len(included))
            for shade, (x, y, w, h) in zip(shades, included):
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, int(shade), 0), 2)
        frame_resized = image_resize(frame_copy, height=img_with_pos.shape[0])

        to_show = np.hstack([frame_resized, img_with_pos])

        cv2.imshow("EYE TRACKER", to_show)



    error_print(error)
    c = cv2.waitKey(1)
    if c == 27:
        break


#cv2.imshow('my image',original_img)
cv2.destroyAllWindows()


import time
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import load_model
from imutils import face_utils

#loading pretrained model
landmark_pred = "weights/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_pred)
model = load_model('weights/drowsyv3.hd5')
IMG_SIZE = (24,24)

def get_eyes(features):
    left_eye=features[36:42]
    right_eye=features[42:48]
    return left_eye,right_eye

def get_mouth(features):
    top_lip=np.vstack([features[50:53],features[61:64]])
    bottom_lip= np.vstack([features[65:68],features[56:60]])
    return top_lip,bottom_lip

def mouth_open(features):
    top_lip,bottom_lip=get_mouth(features)
    bottom_lip_center = np.mean(bottom_lip, axis=0)
    top_lip_center = np.mean(top_lip, axis=0)

def reshape_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 2.3
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
    eye_img = cv2.resize(eye_img, dsize=IMG_SIZE)
    eye_img = eye_img.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    return eye_img

def get_prediction_value(eye_input):
    prediction=model.predict(eye_input)
    return prediction

def predict(left_eye,right_eye):
    left_eye_prediction=get_prediction_value(left_eye)
    right_eye_prediction=get_prediction_value(right_eye)

    prediction=(left_eye_prediction+right_eye_prediction)/2.0
    # print(prediction,end=",")
    print(prediction)
    if prediction<0.85:
        prediction="open"
    else:
        prediction="close"
    print(prediction)
    return prediction

class DrowsyDetector(object):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def __del__(self):
        self.cam.release()

    def get_frame(self):
        close,open = 0,0
        ret,image = self.cam.read()
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        if len(faces)==0:
            eye_state_cnn = 'open'
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            shape = predictor(gray, rect)
            facial_features = face_utils.shape_to_np(shape)

            left_eye_ear,right_eye_ear = get_eyes(facial_features)

            mouth_width=mouth_open(facial_features)

            right_eye_cnn = reshape_eye(gray, eye_points=right_eye_ear)
            left_eye_cnn = reshape_eye(gray, eye_points=left_eye_ear)
            eye_state_cnn = 'open'
            eye_state_cnn = predict(left_eye_cnn,right_eye_cnn)
            # print("Result for face",1+i,":",eye_state_cnn)

            #Drawing circles around the eyes
            for (x, y) in left_eye_ear:
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            for (x, y) in right_eye_ear:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            #Drawing circles around lips
            top_lip,bottom_lip=get_mouth(facial_features)
            for (x, y) in top_lip:
                cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
            for (x, y) in bottom_lip:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            if eye_state_cnn=="close":
                close+=1
                cv2.putText(image, "Eyes: "+eye_state_cnn, (bX,bY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),(0, 0, 255), 2)
            else:
                open+=1
                cv2.putText(image, "Eyes: "+eye_state_cnn, (bX,bY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.rectangle(image, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 2)

        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', image)
        return eye_state_cnn, jpeg.tobytes()

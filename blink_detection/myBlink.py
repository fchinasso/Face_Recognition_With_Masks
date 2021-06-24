import cv2
import dlib
#import math
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0


EAR_THRESHOLD = 0.2
FRAME_THRESHOLD = 2
FRAME_COUNTER = 0
TOTAL_BLINKS = 0

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


while True:

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(round(fps,3))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    

    Vcheck, frame = cap.read()

    if not Vcheck:
        print("Can't receive frame. Exiting ...")
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    tic = time.perf_counter()
    faces = detector(grayFrame, 0)
    toc = time.perf_counter()
    print(f"Time to locate face:{toc - tic:.4f}")

    for face in faces:

        tic = time.perf_counter()
        landmarks = predictor(grayFrame, face)
        landmarks = face_utils.shape_to_np(landmarks)
        toc = time.perf_counter()
        print(f"Time to predict landmarks:{toc - tic:.4f}")

        leftEye = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
   
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
  

        if ear < EAR_THRESHOLD:
            FRAME_COUNTER += 1
            #print (ear)
        
        else:

            if FRAME_COUNTER >= FRAME_THRESHOLD:
                TOTAL_BLINKS += 1

            FRAME_COUNTER = 0
    
        
    cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINKS), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame,"FPS: {}".format(fps),(505,470),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break

cv2.destroyAllWindows()




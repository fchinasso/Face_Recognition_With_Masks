import face_recognition 
import numpy as np
import cv2
import math

CONFIDENCE = 0.6

class face_detector():
    def __init__(self):
        # Net architecture
        prototxt_path = "face_detector/deploy.prototxt"
        # Net weight
        caffemodel_path = "face_detector/weights.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        

    def return_box(self,image):
        (h, w) = image.shape[:2]
        # prepares image for entrance on the model
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # put image on model
        self.detector.setInput(blob)
        # propagates image through model
        detections = self.detector.forward()
        """"
        detections,  4 columms which are:
        0 column -->
        1st column -->
        2nd column --> number of detections, by default 200 (?)
        3th column --> 7 subcollumns which are
            4.0 -->
            4.1 -->
            4.2 --> confidence
            4.3 --> x0
            4.4 --> y0
            4.5 --> x1
            4.6 --> y1
        """
        # check confidance of 200 predictions
        list_box = []
        for i in range(0, detections.shape[2]):
            # box --> array[x0,y0,x1,y1]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # confidence range --> [0-1]
            confidence = detections[0,0,i,2]
            
            if confidence >=CONFIDENCE:
                if list_box == []:
                    list_box = np.expand_dims(box,axis=0)
                else:
                    list_box = np.vstack((list_box,box))

        return list_box

    def detect_face(self,image):
        '''
        Input: imagen numpy.ndarray, shape=(W,H,3)
        Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla represents an detected face
        si no se detecta nada  --> Output: []
        '''
        list_box = self.return_box(image)
        try:
            box_faces = [(box[1],box[2],box[3],box[0]) for box in list_box.astype("int")]
        except:
            box_faces = []
        return box_faces

    def bounding_box_area(self,box):

        try:         
            area = math.sqrt(pow((box[3]-box[1]),2))*math.sqrt(pow((box[2]-box[0]),2))

        except Exception as e: 
                print(e)
                return
        
        return area




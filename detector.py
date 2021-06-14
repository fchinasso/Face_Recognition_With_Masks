import face_recognition 
import numpy as np
import cv2


class face_detector():
    def __init__(self):
        # arquitectura de la red
        prototxt_path = "face_detector/deploy.prototxt"
        # pesos de la red
        caffemodel_path = "face_detector/weights.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        print("passou")

    def return_box(self,image):
        (h, w) = image.shape[:2]
        # preparo la imagen para ingresar al modelo
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # ingreso la imagen al modelo
        self.detector.setInput(blob)
        # propago la imagen hacia adelante del modelo
        detections = self.detector.forward()
        """"
        detections, tiene 4 columnas que son:
        0 column -->
        1st column -->
        2nd column --> numero dedetecciones que hizo por defecto 200
        3th column --> tiene 7 subcolumnas que son
            4.0 -->
            4.1 -->
            4.2 --> confidence
            4.3 --> x0
            4.4 --> y0
            4.5 --> x1
            4.6 --> y1
        """
        # reviso la confianza de las 200's predicciones
        list_box = []
        for i in range(0, detections.shape[2]):
            # box --> array[x0,y0,x1,y1]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # confidence range --> [0-1]
            confidence = detections[0,0,i,2]
            if confidence >=0.6:
                if list_box == []:
                    list_box = np.expand_dims(box,axis=0)
                else:
                    list_box = np.vstack((list_box,box))
        
        return list_box

    def detect_face(self,image):
        '''
        Input: imagen numpy.ndarray, shape=(W,H,3)
        Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla representa un rostro detectado
        si no se detecta nada  --> Output: []

        antes -->    box_faces = face_recognition.face_locations(image)
        '''
        list_box = self.return_box(image)
        try:
            box_faces = [(box[1],box[2],box[3],box[0]) for box in list_box.astype("int")]
        except:
            box_faces = []
        return box_faces



from numpy import ComplexWarning, byte
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import time
import detector 
import math
import communications
from enum import Enum
from datetime import datetime

#Verbose for Debug
verbose=True

#Dirs and extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TRAIN_DIR = "photos/known"
TEST_DIR = "photos/unknown"

#Initiates detector
rec_face = detector.face_detector()

#Initiates Serial Handler
serial= communications.SerialHandler(verbose)

#Proportion factor to start a matching 
PROPORTION_FACTOR = 5.5


#Random color generator
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


#Trains Classifier
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    X = []
    Y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):

            image = face_recognition.load_image_file(img_path)
            #Uses previously iniate detector 
            face_bounding_boxes = rec_face.detect_face(image=image)


            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                Y.append(class_dir)
                

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, Y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

#Given an image gives prediction
def predict(X_img_path_or_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
  
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)


    # Load image file and find face locations
    if os.path.isfile(X_img_path_or_frame):
        X_img = face_recognition.load_image_file(X_img_path_or_frame)
    
    if  not os.path.isfile(X_img_path_or_frame):
        X_img = X_img_path_or_frame

    #X_face_locations = face_recognition.face_locations(X_img)
    X_face_locations = rec_face.detect_face(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

#Displays labels on image
def show_prediction_labels_on_image(img_path, predictions):
   
    image = face_recognition.load_image_file(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
   
    for name, (top, right, bottom, left) in predictions:
        
        top_right = (right, top)
        bottom_left = (left, bottom + 22)
        bottom_right=(right,bottom)
        a = left
        b = bottom-top
        top_left=(top,left)
        cv2.rectangle(image, top_right,bottom_left, (255,0,0), 3)
        cv2.putText(image, str(name), (left,bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, name_to_color(name),1,cv2.FILLED)
    
    cv2.imshow(img_path, image)
    cv2.waitKey(0)
    cv2.destroyWindow(img_path)

def Matching_Debug():

    cap = cv2.VideoCapture(0)
        
    width = cap.get(3)
    height = cap.get(4)
    frame_area = width * height

    print(height)
    print(width)

    while True:
            
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        bounding_boxes = rec_face.detect_face(frame)
   
        for faces in bounding_boxes:

            #[(y0,x1,y1,x0)]

            top_right = (faces[1], faces[0])
            bottom_left = (faces[3], faces[2])
    

            area = rec_face.bounding_box_area(faces)

            if(faces[2] < height and area > frame_area/PROPORTION_FACTOR):
    
                predictions = predict(frame, model_path="trained_knn_model.clf")

                for name, (top, right, bottom, left) in predictions:
                
                    #print(predictions)
                    if(bottom < height and right < width):
                        print(f"Found:{name},Top:{top},Right:{right},Left:{left},Botoom:{bottom}")
                        serial.send_matching(name)
                            

            cv2.circle(frame,top_right, 10, (0,0,255), -1)
            cv2.circle(frame,bottom_left, 10, (0,0,255), -1)
            cv2.rectangle(frame, top_right,bottom_left, (255,0,0), 3)
            cv2.putText(frame,"Area:{}".format(area),(10,int(height-10)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
      
        serial.pooling()
        if serial.W_State == communications.W_States.Maintenence:
            cv2.destroyAllWindows()
            cap.release()
            return

def Matching():

    cap = cv2.VideoCapture(0)
    width = cap.get(3)
    height = cap.get(4)
    frame_area = width * height
    print(frame_area)

    while True:
            
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        bounding_boxes = rec_face.detect_face(frame)
   
        for faces in bounding_boxes:

            #[(y0,x1,y1,x0)]
    
            area = rec_face.bounding_box_area(faces)

            if(faces[2] < height and area > frame_area/PROPORTION_FACTOR):
    
                predictions = predict(frame, model_path="trained_knn_model.clf")

                for name, (top, right, bottom, left) in predictions:                
                    #print(predictions)
                    if(bottom < height and right < width):
                        print(f"Found:{name},Top:{top},Right:{right},Left:{left},Botoom:{bottom}")
                        serial.send_matching(name)
      
        serial.pooling()

        if serial.W_State != communications.W_States.Recognition:
            cap.release()
            return
                        
def train_predict():

    tic = time.perf_counter()
    print("Training KNN classifier...")
    #Creates Classifier
    classifier = train(TRAIN_DIR, model_save_path="trained_knn_model.clf",verbose=verbose)
    print("Training complete!")
    toc = time.perf_counter()

    if verbose:
       print(f"Time to train {toc - tic:0.4f} seconds")

    
    for image_file in os.listdir(TEST_DIR):
        full_file_path = os.path.join(TEST_DIR, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        #print(predictions)

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join(TEST_DIR, image_file), predictions)


if __name__ == "__main__":
   
   while True:
       
        if serial.W_State == communications.W_States.Maintenence:
            serial.pooling()
            

        if serial.W_State == communications.W_States.Recognition:
            if verbose:
                print("Entered Recognition routine")
            Matching_Debug()
#            Matching()

        if serial.W_State == communications.W_States.Training:
            print("Training")
            if verbose:
                print("Entered training routine")
            train(TRAIN_DIR, model_save_path="trained_knn_model.clf",verbose=True)

            #Returns to Maintenence 
            serial.change_state("02020100",8)
            
            
            
       
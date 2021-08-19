import math
import os
import shutil
import time

import cv2 as cv
import serial

import detector

USERS_DIR = "photos/known"
IMAGE_PER_USER = 4
TIME_BETWEEN_PHOTOS = 3

#Initiates detector
rec_face = detector.face_detector()

class SerialHandler():

    def __init__(self,Verbose=False):
        self.serial=serial.Serial("/dev/ttyACM1",baudrate=115200, parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS,timeout=2)
        self.Verbose = Verbose
        self.cap = cv.VideoCapture(0)
        
         

    def pooling(self):

        while self.serial.in_waiting:
           
            message = self.serial.read(20).hex()
            if self.Verbose:
                print(f"Message:{message}")
            self.parse(message)
            

    def parse(self,message):
    
        if len(message)>0:

            try:
                m_header = message[0]+message[1]
                m_lenght = (int(message[2]+message[3], base=16))
                m_type = message[4] + message [5]
                if self.Verbose:
                    print(f"Header:{m_header}")
                    print(f"Lenght: {m_lenght}")
                    print(f"Type: {m_type}")
                    
            
            except:
                print("ERROR:Failed to parse.")
                return

            #Register new user on bank
            if m_header == "02" and m_type == "03":
                self.register_new_user(message,m_lenght)

            if m_header == "02" and m_type == "09":
                self.exclude_user(message,m_lenght)

    
    def register_new_user(self,message,m_lenght):
        
        user = ""
        register_size = 2*(m_lenght-2)


        for x in range(register_size):
            
            user = user + message[8+x]


        user_hex = user
        user = bytearray.fromhex(user).decode()

        if self.Verbose:
            print(f"Creating new user:{user}")
        
        user_dir=os.path.join(USERS_DIR,user)

        if os.path.exists(user_dir):
            print("ERROR:User already exists.")
            return

        else:
            try:

                os.mkdir(user_dir)
                
                start_time = time.time()
                timeout_time = time.time()
                img_counter = 0
                
                while img_counter < IMAGE_PER_USER:

                    if time.time()-timeout_time >= TIME_BETWEEN_PHOTOS*IMAGE_PER_USER*2:
                        print(f"WARNING:Timeout to take pictures, {img_counter} were taken.")
                        break

                    m_tosend = "03" + '{0:x}'.format(m_lenght).zfill(2) + "03" + '{0:x}'.format(img_counter+1).zfill(2) + user_hex

                    ret,frame = self.cap.read()
                    
                    if time.time() - start_time >= TIME_BETWEEN_PHOTOS: 

                        bounding_boxes = rec_face.detect_face(image=frame)

                        if len(bounding_boxes) == 1:
                            
                            cv.imwrite(f"{os.path.join(user_dir,str(img_counter))}.jpg",frame)

                            if self.Verbose:
                                print(f"Image {img_counter} written.")
                                print(m_tosend)
                            start_time = time.time()
                            img_counter += 1

                    
                        else:
                            print("Photo not suitable for training: {}".format("Didn't find a face" if len(bounding_boxes) < 1 else "Found more than one face."))
                            start_time = time.time()

                    #self.serial.write(m_tosend.encode('utf-8'))
                self.cap.release()

            except:
                print("ERROR:Failed to take pictures.")
                return


    def exclude_user(self,message,m_lenght):

        user = ""
        register_size = 2*(m_lenght-2)


        for x in range(register_size):
            
            user = user + message[8+x]

        user = bytearray.fromhex(user).decode()


        if self.Verbose:
            print(f"Deleting user:{user}")
        
        user_dir=os.path.join(USERS_DIR,user)

        if not os.path.exists(user_dir):
            print("ERROR:User doesn't exists.")
            m_tosend = "03" + "02" + "09" + "02"
                      

        else:
            try:
                shutil.rmtree(user_dir)
                if self.Verbose:
                    print(f"User:{user} deleted.")
                    m_tosend = "03" + "02" + "09" + "01"
                
                
            except OSError as e:
                print(f"ERROR: {e.filename}- {e.strerror}.")
                m_tosend = "03" + "02" + "09" + "03"
        
        print(m_tosend)
        return




                
                

                       
                                                                                               

    
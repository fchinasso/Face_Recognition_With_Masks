import math
import os
from re import M
import shutil
import time
from enum import Enum
import cv2 as cv
import serial
import detector

USERS_DIR = "photos/known"
IMAGE_PER_USER = 4
TIME_BETWEEN_PHOTOS = 3

#Initiates detector
rec_face = detector.face_detector()


#Enum of Working States 
class W_States(Enum):
    Maintenence = 1
    Recognition = 2
    Training = 3


#Enum of Serial States 
class S_States(Enum):
    Working = 1
    Unitialized = 2

class SerialHandler():

    def __init__(self,Verbose=False):

        self.Verbose = Verbose

        try:
            self.serial=serial.Serial("/dev/ttyACM0",baudrate=115200, parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS,timeout=2)
            if self.Verbose:
                print("Serial Initialized")
            self.S_State = S_States.Working

        except Exception as e:
            self.S_State = S_States.Unitialized 
            print(e)

        self.W_State = W_States.Recognition
       

    def pooling(self):

        if self.S_State == S_States.Working:
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

            if self.W_State == W_States.Maintenence: 

                #Register new user on bank
                if m_header == "02" and m_type == "03":
                    self.register_new_user(message,m_lenght)

                #Exclude User from bank
                if m_header == "02" and m_type == "09":
                    self.exclude_user(message,m_lenght)

                if m_header == "02" and m_type == "01":
                    self.change_state(message,m_lenght)   

            if self.W_State == W_States.Recognition:

                if m_header == "02" and m_type == "01":
                    self.change_state(message,m_lenght)


    
    def register_new_user(self,message,m_lenght):

        self.cap = cv.VideoCapture(0)

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
            self.cap.release()
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

                            m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                            m_tosend=bytearray.fromhex(m_tosend).decode("ISO-8859-1")
                 
                            if self.Verbose:
                                print(f"mtosend:{m_tosend}")

                            self.serial.write(m_tosend.encode())   

                            start_time = time.time()
                            img_counter += 1

                        else:
                            print("Photo not suitable for training: {}".format("Didn't find a face" if len(bounding_boxes) < 1 else "Found more than one face."))
                            start_time = time.time()
                            
                self.cap.release()

                if img_counter < IMAGE_PER_USER/2:

                    if(self.Verbose):
                        print(f"Failed to take enough pictures,{img_counter} pics were taken.")

                    shutil.rmtree(user_dir)
                    m_tosend = "03" + "02" + "03" + "0b"
                    m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                    m_tosend=bytearray.fromhex(m_tosend).decode("ISO-8859-1") 
                    self.serial.write(m_tosend.encode())


                if img_counter >= IMAGE_PER_USER/2:
                    if (self.Verbose):
                        print(f"Succefully registered user {user},{img_counter} pics were taken.")
                    m_tosend = "03" + "02" + "03" + "0c"
                    m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                    m_tosend=bytearray.fromhex(m_tosend).decode("ISO-8859-1") 
                    self.serial.write(m_tosend.encode())


            except Exception as e: 
                print(e)
                print("ERROR:Failed to take pictures.")
                shutil.rmtree(user_dir)
                m_tosend = "03" + "02" + "03" + "00"
                m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                m_tosend=bytearray.fromhex(m_tosend).decode("ISO-8859-1") 
                self.serial.write(m_tosend.encode())
                self.cap.release()
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
            m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
            m_tosend=bytearray.fromhex(m_tosend).decode("ISO-8859-1") 
            self.serial.write(m_tosend.encode())
            return
                      

        else:
            try:
                shutil.rmtree(user_dir)
                if self.Verbose:
                    print(f"User:{user} deleted.")
                    m_tosend = "03" + "02" + "09" + "01"
                    m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                    m_tosend=bytearray.fromhex(m_tosend).decode() 
                    self.serial.write(m_tosend.encode())
                
                
            except OSError as e:
                print(f"ERROR: {e.filename}- {e.strerror}.")
                m_tosend = "03" + "02" + "09" + "03"
                m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                m_tosend=bytearray.fromhex(m_tosend).decode() 
                self.serial.write(m_tosend.encode())
        
        return


    def change_state(self,message,m_lenght):

        mode = message[6]+message[7]
        
        if mode == "00":
            self.W_State = W_States.Maintenence
            if self.Verbose:
                print("Serial State changed to Maintence")
            
        if mode == "02":
            self.W_State = W_States.Recognition
            if self.Verbose:
                print("Serial State Changed to Recognition")

        if mode == "01":
            self.W_State = W_States.Training
            if self.Verbose:
                print("Serial State Changed to Training")

    def send_matching(self,message):

        if self.S_State == S_States.Unitialized:
           
            if self.Verbose:
                print("ERROR:Serial is not initialized")
            return
        
        if self.S_State == S_States.Working:

          try:

                if message == "unknown":
                    m_tosend = "02010205"

                else:
                    message= message.encode().hex()
                    m_tosend = "03" + "09" + "08" + message

                m_tosend = m_tosend + self.calculate_cheksum(m_tosend)
                m_tosend=bytearray.fromhex(m_tosend).decode("ISO-8859-1") 
                
                self.serial.write(m_tosend.encode())
          
          except Exception as e:
                print(f"ERROR: {e.filename}- {e.strerror}.")

    def calculate_cheksum(self,message):

        lenght = int(message[2] + message[3],16)
        sum = 0
        
        for x in range(2,(lenght+2)*2,2):
            sum = sum + int(message[x:x+2],16)

        if sum > 255:
            sum = sum & 0xff

        sum = '{0:x}'.format(sum).zfill(2) 
        return sum
            


        

    

        
                
                

                       
                                                                                               

    
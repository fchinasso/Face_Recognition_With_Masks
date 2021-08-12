import serial
import os
import cv2 as cv

USERS_DIR = "photos/known"

class SerialHandler():

    def __init__(self,Verbose=False):
        self.serial=serial.Serial("/dev/ttyACM0",baudrate=115200,timeout=0.5)
        self.Verbose= Verbose
        
         

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
                print("Failed to parse")

            #Register new user on bank
            if m_header == "02" and m_type == "03":

                user = ""
                register_size = 2*(m_lenght-2)

                for x in range(register_size):
                    
                    user = user + message[8+x]

                if self.Verbose:
                    print(f"Creating new user:{user}")
                
                dir=os.path.join(USERS_DIR,user)

                if os.path.exists(dir):
                    print("ERROR:User already exists")
                    return

                else:
                    os.mkdir(dir)
                
                

                       
                                                                                               

        


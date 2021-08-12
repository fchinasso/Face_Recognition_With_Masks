import serial

message=[]

class SerialHandler():

    def __init__(self):
        self.serial=serial.Serial("/dev/ttyACM0",baudrate=115200,timeout=0.5)
         

    def pooling(self):

        while self.serial.in_waiting:
            message = self.serial.readline().hex()
            print (message)

__author__ = 'peeyush yadav'

import cv2
from faceRecognition import recognise_face_continuous

integrated_webCam_port = 0
           
def checkUser_available_continuous():   
    camera = cv2.VideoCapture(integrated_webCam_port) #initialise Camera Object
    recognise_face_continuous(camera)
    del(camera)
     
def release_Camera():
    camera = cv2.VideoCapture(0) #initialise Camera Object       
    del(camera)
        
if __name__ == "__main__":
    #release_Camera() # FIx : Run this function if camera doesnt turn off.
    checkUser_available_continuous()

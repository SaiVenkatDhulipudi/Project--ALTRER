from notifypy import Notify
from pyaudio import *
import speech_recognition as sr
import text2emotion as te
import cv2 as c
import numpy as np
from PIL import Image
import os
import pyttsx3 as p
import matplotlib.pyplot as plt
from deepface import DeepFace
#=============================
r = sr.Recognizer()
engine = p.init()
en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
engine.setProperty('voice', en_voice_id)
engine.setProperty('rate', 178)

def speak(audio):
    print(f"{audio}")
    engine.say(audio)
    engine.runAndWait()
def record_audio():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source,0.5)
        r.pause_threshold=1
        audio=r.listen(source)
        voice_data=''
        try:
            voice_data=r.recognize_google(audio,language='en-in')
            print(voice_data)
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass

        return voice_data
def push_notification(me,w):
    hr=Notify()
    hr.title=w
    hr.message=me
    hr.application_name='ALTERER'
    hr.icon = "C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\warning.png"
    hr.send(block=False)
def respond(s):
    keys=['Facebook','WhatsApp','Snapchat','streak','Twitter','tweets','story' , 'post' , 'likes', 'Instagram' , 'status' , 'commented' , 'share']
    for i in keys:
        if(i in s):
            push_notification("YOU HAVE LOT OF LIST TO DO","warning")
            return
def recog():
    recogniser=c.face.LBPHFaceRecognizer_create()
    recogniser.read('C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\trainer/trained.yml')
    cascadepath= 'C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\haarcascade_frontalface_default.xml'
    facecascade= c.CascadeClassifier(cascadepath)
    font=c.FONT_HERSHEY_SCRIPT_SIMPLEX
    id=2
    names=['','SV']
    cam=c.VideoCapture(0,c.CAP_DSHOW)
    cam.set(3,640)
    cam.set(4,480)
    minw=0.1*cam.get(3)
    minh=0.1*cam.get(4)
    while True:
        ret,img=cam.read()
        converted_image=c.cvtColor(img,c.COLOR_BGR2GRAY)
        faces=facecascade.detectMultiScale(
        converted_image, 
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minw),int(minh)),
        )
        i=0
        for (x,y,w,h) in faces:
            c.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            id,accuracy=recogniser.predict(converted_image[y:y+h,x:x+w])
            i=i+1
            if(accuracy<70 and accuracy>30):
                plt.imshow(c.cvtColor(img,c.COLOR_BGR2RGB))
                p=DeepFace.analyze(img,actions=['emotion'])
                if (p['dominant_emotion']=='angry'):
                    speak("HEY BUDDY WHAT HAPPENED? WHY ARE YOU LOOKING ANGRY?")
                
            else :
                speak("ALTRER ONLY MEANT FOR WHO WANTS TO BE PRODUCTIVE")



if __name__=="__main__":
    speak("recognising")
    recog()
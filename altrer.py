from sre_parse import SPECIAL_CHARS
from notifypy import Notify
import speech_recognition as sr
import cv2 as c
import numpy as np
import os
import pyttsx3 as ps
from deepface import DeepFace
import matplotlib.pyplot as plt
from threading import *
from datetime import date,datetime
import threading as th
import csv
#=============================
r = sr.Recognizer()
def speak(audio):
    engine = ps.init()
    en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty('voice', en_voice_id)
    engine.setProperty('rate', 158)
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
# def writer(p):
#     fields = ['Date','Time', 'Emotions',*p.keys(),'distract_count']
#     filename = "C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\emotions.csv"
#     with open(filename, 'w') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(fields) 
#         csvwriter.writerows(rows)
#         csvfile.close()
def push_notification(me,w):
    hr=Notify()
    hr.title=w
    hr.message=me
    hr.application_name='ALTERER'
    hr.icon = "C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\warning.png"
    hr.send(block=False)
            
def voice():
    global distract_count
    while True:
        t=record_audio()
        keys=['Facebook','WhatsApp','Snapchat','streak','Twitter','tweets','story' , 'post' , 'likes', 'Instagram' , 'status' , 'commented' , 'share']
        res=list(filter(lambda x: x in keys,t.split()))
        if len(res):
            distract_count+=1
            audio=("YOU HAVE LOT OF LIST TO DO")
            speak(audio)
            push_notification(audio,'Warning')       
def recog():
    recogniser=c.face.LBPHFaceRecognizer_create()
    recogniser.read('C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\trainer/trained.yml')
    cascadepath= 'C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\haarcascade_frontalface_default.xml'
    facecascade= c.CascadeClassifier(cascadepath)
    id=2
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
                p=dict()
                try:
                    p=DeepFace.analyze(img,actions=['emotion'])
                    print(p)
                except:
                    pass
                if len(p):
                    # now = datetime.now()
                    # current_time = now.strftime("%H:%M:%S")
                    # today = date.today()
                    # rows.append([today,current_time,*p.values(),distract_count])
                    # k=c.waitKey(100)&0xff
                    # if k==27:
                    #     writer(p)
                    #     exit()
                        
                        
                    if (p['dominant_emotion']=='angry'):
                        speak("HEY BUDDY WHAT HAPPENED? WHY ARE YOU LOOKING ANGRY?")
                    elif (p['dominant_emotion']=='sad'):
                        speak("HEY BUDDY WHAT HAPPENED? WHY ARE YOU SO SAD?")
                    elif (p['dominant_emotion']=='fear'):
                        speak("HEY BUDDY WHAT HAPPENED? what bothers you")
                    elif (p['dominant_emotion']=='disgust'):
                        speak("HEY BUDDY WHAT HAPPENED")
                    elif (p['dominant_emotion']=='happy'):
                        speak("that's the spirit!!")
                    print(p['dominant_emotion'])
            # else :
            #     # if i<3:
            #     #     speak("ALTRER ONLY MEANT FOR WHO WANTS TO BE PRODUCTIVE")
            #     pass

if __name__=="__main__":
    speak("recognising")
    rows=[]
    distract_count=0
    t2=th.Thread(target=recog)
    t2.start()
    voice()
    


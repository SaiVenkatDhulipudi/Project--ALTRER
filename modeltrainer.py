import cv2 as c
import numpy as np
from PIL import Image
import os
#----------
path='C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\samples'
recogniser=c.face.LBPHFaceRecognizer_create()
detector=c.CascadeClassifier('C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\haarcascade_frontalface_default.xml')
def image_and_labels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    facesamples=[]
    ids=[]
    for imagepath in imagePaths:
        gray_img=Image.open(imagepath).convert('L')
        img_arr=np.array(gray_img,'uint8')
        id=int(os.path.split(imagepath)[-1].split(".")[1])
        faces=detector.detectMultiScale(img_arr)
        for(x,y,w,h) in faces:
            facesamples.append(img_arr[y:y+h,x:x+w])
            ids.append(id)
    return facesamples,ids
print("training finished wait for a while")

faces,ids=image_and_labels(path)
recogniser.train(faces,np.array(ids))
recogniser.write('C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\trainer/trained.yml')
print("Model trained,now I can recognise your face")

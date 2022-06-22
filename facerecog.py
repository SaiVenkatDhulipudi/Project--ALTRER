import cv2 as c
#--------------------
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
        i+=1
        if(accuracy<50 and accuracy>30):
            print("recognised"+str(" {0}%".format(round(100-accuracy))))
        else:
            print("not recognised")
    k=c.waitKey(10)&0xff
    if k==27 or i==9:
        break
print("HAVE A GOOD DAY")
cam.release()
c.destroyAllWindows()

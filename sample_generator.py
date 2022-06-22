import cv2 as c
cam=c.VideoCapture(0,c.CAP_DSHOW)
cam.set(3,640)
cam.set(4,480)
detector= c.CascadeClassifier('C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\haarcascade_frontalface_default.xml')
faceid=input("enter sample no")
print("taking samples look at camera")
count=0
while True:
    ret,img=cam.read()
    converted_image=c.cvtColor(img,c.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(converted_image, 1.3, 5)
    for (x,y,w,h) in faces:
        c.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        c.imwrite("C:\\Users\\sai venkat dhulipudi\\Documents\\python 2021\\altrer\\samples/face."+str(faceid)+'.'+str(count)+".jpg",converted_image[y:y+h,x:x+w])
        c.imshow('image',img)
    k=c.waitKey(100)&0xff
    if k==27 :
        break
    elif count>=50 :
        break
print("samples taken close the program")
cam.release()
c.destroyAllWindows()
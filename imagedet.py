import cv2 as c
from matplotlib.colors import cnames
import numpy as np
 #====
net =c.dnn.readNet('yolov3.weights','yolov3.cfg' )
classes=[]
with open('coco.names','r') as f :
    classes=f.read().splitlines()
    cam=c.VideoCapture(0,c.CAP_DSHOW)
    cam.set(3,640)
    cam.set(4,480)
    minw=0.1*cam.get(3)
    minh=0.1*cam.get(4)
    while True:
        ret,img=cam.read()
        he,wi,_=img.shape()
        blob=c.blobFromImage(img,1/255,(416,416)(0,0,0),swapRb=True,crop=False)
        net.setInput(blob)
        output_layers_names=net.getUnconnectedOutLayersNames()
        layeroutputs=net.forward(output_layers_names)
        boxes=[]
        confidences=[]
        class_ids=[]
        for i in layeroutputs:
            for d in i:
                scores=d[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]
                if confidence>0.5:
                    center_x=int(d[0]*wi)
                    center_y=int(d[1]*he)
                    w=int(d[2]*wi)
                    h=int(d[3]*he)
                    x=int(center_x-w/2)
                    y=int(center_y-h/2)
                    boxes.append({x,y,w,h})
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes=c.dnn.NMSBoxes(boxes,confidences,0.5,0.4)   
        font=c.FONT_HERSHEY_COMPLEX
        colors=np.random.uniform(0.255,size=(len(boxes),3))
        for i in indexes.flatten()  :
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            con=str(round())
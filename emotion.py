import cv2 as c
from deepface import DeepFace
import matplotlib.pyplot as plt
img=c.imread("C:\\Users\\sai venkat dhulipudi\\Desktop\\svchildhood.jpg")
plt.imshow(c.cvtColor(img,c.COLOR_BGR2RGB))
p=DeepFace.analyze(img)
print(p)
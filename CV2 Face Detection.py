import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


data_path="./Capture Image/"
onlyfiles =[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data,Labels=[],[]


for i,files in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images =cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels=np.asarray(Labels,np.int32)


model=cv2.face.LBPHFaceRecognizer_create()#LBPHFaceRecognize stand for:--> Local Binary Patterns Histograms Face Recognizer


model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Has Been Trained ")



face_clasifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clasifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    if faces is():
        return img,[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # roi=Region of Intrest
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi

cap=cv2.VideoCapture(0)

while True:
    _,Frame=cap.read()

    image,face=face_detector(Frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            Display=str(confidence)+"% Matching"
        cv2.putText(image,Display,(100,200),cv2.FONT_HERSHEY_SIMPLEX,1,(250,120,255),2)
        if confidence>75:
            cv2.putText(image,"Unlock",(250,450),cv2.FONT_HERSHEY_SIMPLEX,1,(250,120,255),2)
            cv2.imshow('Face Cropper',image)
        else:
            cv2.putText(image,"locked",(250,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',image)

    except:
        cv2.putText(image,"Face Not Found",(250,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow('Face Cropper',image)
        pass
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()
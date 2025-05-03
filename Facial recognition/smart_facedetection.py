import cv2
import numpy as np
import face_recognition as fr
import os


def resize_image(image, target_size=(640, 480)):
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized

path = r'C:\Users\ronjd\OneDrive\Desktop\comp_V\Facial recognition\images'
image=[]
image_name=[]

myList= os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    image_name.append(os.path.splitext(cl)[0])
   
print(image_name)

#find face location and encodings

def find_encodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encode_listKnown = find_encodings(image)

vid= cv2.VideoCapture(0)

while True:
    success, frame = vid.read()
    frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    faces_in_frame = fr.face_locations(frames)
    encodes_in_frame = fr.face_encodings(frames, faces_in_frame)

    for encodeFace, faceLoc in zip(encodes_in_frame, faces_in_frame):
        matches = fr.compare_faces(encode_listKnown, encodeFace)
        faceDis = fr.face_distance(encode_listKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = image_name[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
         break
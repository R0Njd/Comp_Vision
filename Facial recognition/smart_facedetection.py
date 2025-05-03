import cv2
import numpy as np
import face_recognition as fr
import os
import time

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

frame_count = 0
process_every_n_frames = 3  # Only process every 3 frames

while True:
    success, frame = vid.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if frame_count % process_every_n_frames == 0:
        faces_in_frame = fr.face_locations(rgb_small_frame)
        encodes_in_frame = fr.face_encodings(rgb_small_frame, faces_in_frame)

        names = []
        for encodeFace in encodes_in_frame:
            matches = fr.compare_faces(encode_listKnown, encodeFace)
            faceDis = fr.face_distance(encode_listKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            name = "Unknown"

            if matches[matchIndex]:
                name = image_name[matchIndex].upper()
            names.append(name)

    for (faceLoc, name) in zip(faces_in_frame, names):
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('webcam', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
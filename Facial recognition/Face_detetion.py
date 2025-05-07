#import libraries
import cv2
import numpy as np
import face_recognition as fr



def resize_image(image, target_size=(640, 480)):
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized


#imaport images 
img1 = fr.load_image_file(r"C:\Users\ronjd\OneDrive\Desktop\comp_V\Facial recognition\images\image2.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1_resized = resize_image(img1)
img2 = fr.load_image_file(r"C:\Users\ronjd\OneDrive\Desktop\comp_V\Facial recognition\images\image4.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2_resized = resize_image(img2)

print(img1_resized.shape)
print(img2_resized.shape)

#find face location and encodings
face_loc1 = fr.face_locations(img1_resized)[0]
encode_loc1 = fr.face_encodings(img1_resized)[0]
cv2.rectangle(img1_resized, (face_loc1[3], face_loc1[0]), (face_loc1[1], face_loc1[2]), (255, 0, 255), 2)
face_loc2 = fr.face_locations(img2_resized)[0]
encode_loc2 = fr.face_encodings(img2_resized)[0]  
cv2.rectangle(img2_resized, (face_loc2[3], face_loc2[0]), (face_loc2[1], face_loc2[2]), (255, 0, 255), 2)

#compare faces
results = fr.compare_faces([encode_loc1], encode_loc2)[0]
print(results)
cv2.putText(img2_resized, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("image1", img1_resized)
cv2.imshow("image2", img2_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import sys


face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread(sys.argv[1])


#img = cv2.imread('1.jpg')


faces = face_cascade.detectMultiScale(img, 1.3, 5)

for (x, y, w, h) in faces:
    crop_img = img[y:y+h, x:x+w]
    resized_img = cv2.resize(crop_img, (20, 20))
    cv2.imwrite('20_' + sys.argv[1] , resized_img )
    
cv2.waitKey(0)




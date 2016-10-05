import cv2
import sys



face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')
#img = cv2.imread(sys.argv[1])


img = cv2.imread(sys.argv[1])
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img)
print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    rect_gray_img = gray_img[y:y+h][x:x+w]
    rect_color_img = img[y:y+h][x:x+w]
    
    
    eyes = eyes_cascade.detectMultiScale(rect_gray_img)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(rect_color_img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    smile = smile_cascade.detectMultiScale(rect_color_img, 1.7, 11)
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(rect_color_img, (sx, sy), (sx+sw, sy+sh), (255, 255, 0), 2)

    nose = nose_cascade.detectMultiScale(rect_gray_img)
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(rect_color_img, (nx, ny), (nx+nw, ny+nh), (255, 0, 128), 2)   
    
    cv2.imwrite("res_img.ppm", img)

cv2.imshow("",  img)
cv2.waitKey(0)




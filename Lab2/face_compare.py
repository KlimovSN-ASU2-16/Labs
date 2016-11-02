import cv2
import sys
import math
import os
import random as rand

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')




def main():

    photo_dict =dict()
    value = 0
    os.chdir('./Выборка')
    for i in os.listdir('./'):
        img = cv2.imread(i)
        cv2.imshow(' ',img)
        photo_dict[i] = normalization(calculate_distance(find_centres(img)))
        
        
    colum_names = os.listdir('./')
    colum_names.sort()
    
    print(compare(photo_dict.get('foto1.ppm'), photo_dict.get('foto9.ppm')))
   #print(compare(normalization(photo_dict.get("foto1.ppm")), normalization(photo_dict.get("foto2.ppm"))))
    string = "\t\t"
    for i in colum_names:
       string = string + i + " | "
    print(string)
    
    out_string = ""
    for i in colum_names:
        out_string += i
        for j in colum_names:
            out_string = out_string + "\t" + str(round(compare(photo_dict.get(i),  photo_dict.get(j)), 3)) + ""  + "\t|"
            
        print(out_string)
        out_string = ""
    
    """for i in slovar.keys():
        for j in slovar.keys():
            if (j>i):
                for k in slovar.get(i):
                    ranges += math.sqr(slovar.get(i)[k]-slovar.get(j)[k])
                ranges = math.sqrt(ranges)/2*100
                print('range between', i, ' and ', j,' = ', ranges)"""


def compare (vector1,vector2):
    ranges = 0
    for i in range(len(vector1)):
        ranges += math.pow(vector1[i]-vector2[i], 2)
        #print (vector1[i],' ', vector2[i],' ', math.pow(vector1[i]-vector2[i], 2))
    ranges = abs((math.sqrt(ranges))*100-100)
    return ranges

def calculate_distance(centres):
    
    distances = list()
    for centre in centres:
        j = centres.index(centre)+1
        while j < len(centres):
            distances.append(math.sqrt( math.pow(centre[0] - centres[j][0], 2) + math.pow(centre[0] - centres[j][1], 2)))
            j=j+1
    return distances
    

def normalization(distances):
    norm_dist = list()
    norm = 0
    for distance in distances:
        norm += math.pow(distance,2)
    vector_length = math.sqrt(norm)
    for distance in distances:
        norm_dist.append(distance/vector_length)
    
    return norm_dist
    

            
 
    

def find_centres(img):
    
    
    faces = face_cascade.detectMultiScale(img)
    

    for (x, y, w, h) in faces:
     
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        center_face = (x+w//2, y+h//2)
        cv2.circle(img, center_face, 5, (255, 0, 0), 2)
        
        
        
        indent_x = x
        indent_y = y
        
        upper_left_face = img[y:y+h//2, x:x+w//2]

        left_eyes = eye_cascade.detectMultiScale(upper_left_face, 1.01,  6)
        for (le_x, le_y, le_w, le_h) in left_eyes:
            cv2.rectangle(upper_left_face, (le_x, le_y), (le_x+le_w, le_y+le_h), (255, 255, 0), 2)
            center_left_eye = (indent_x + le_x+le_w//2,indent_y +  le_y+le_h//2)
            cv2.circle(img, center_left_eye, 5, (255, 255, 0), 2)
        

        
        
        indent_x = x+w//2
        indent_y = y
        upper_right_face = img[y:y+h//2, x+w//2:x+w]
        
        right_eyes = eye_cascade.detectMultiScale(upper_right_face, 1.01,  6)
        for (re_x, re_y, re_w, re_h) in right_eyes:
            cv2.rectangle(upper_right_face, (re_x, re_y), (re_x+re_w, re_y+re_h), (255, 255, 0), 2)
            center_right_eye = (indent_x+re_x+re_w//2, indent_y + re_y+re_h//2)
            cv2.circle(img, center_right_eye, 5, (255, 255, 0), 2)

        
        
        
        indent_x = x
        indent_y = y+h//2
        bottom_face = img[y+h//2:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(bottom_face, 1.01,  6)
        smile_rect = get_bigger_rectangle(smiles)
        cv2.rectangle(bottom_face, (smile_rect[0], smile_rect[1] ), (smile_rect[0]+smile_rect[2], smile_rect[1]+smile_rect[3]), (255,128 , 128), 2)
        center_smile = (indent_x + smile_rect[0]+smile_rect[2]//2, indent_y + smile_rect[1]+smile_rect[3]//2)
        cv2.circle(img, center_smile, 5, (255, 128, 128), 2)

        
        
        indent_x = x+(w//2)//2
        indent_y = y+(h//2)//2
        middle = img[y+(h//2)//2:y+h-((h//2)//2), x+(w//2)//2:x+w-(w//2)//2]
        noses = nose_cascade.detectMultiScale(middle, 1.01,  6 )
        nose_rect = get_bigger_rectangle(noses)
        cv2.rectangle(middle, (nose_rect[0], nose_rect[1] ), (nose_rect[0]+nose_rect[2], nose_rect[1]+nose_rect[3]), (255,128 , 128), 2)
        center_nose = (indent_x + nose_rect[0]+nose_rect[2]//2, indent_y + nose_rect[1]+nose_rect[3]//2)
        cv2.circle(img, center_nose, 5, (0, 128, 128), 2)

       
       
    return [center_face,   center_left_eye,  center_right_eye,  center_nose,  center_smile]
    #return dict([("face", center_face), ("left_eye", center_left_eye), ("right_eye", center_right_eye), ("nose", center_nose), ("smile", center_smile)])
        


def get_bigger_rectangle(rectangles):
    
    area = 0
    bigger_rect = rectangles[0]
    for rect in rectangles:
        if(rect[2]*rect[3]>area):
            area=rect[2]*rect[3]
            bigger_rect=rect
                
    return bigger_rect        
        
    
    
        
    
if __name__=="__main__":
    main()




import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def main():

    img1 = cv2.imread('iron.jpg')         
    img2 = cv2.imread('iron_1.jpg') 
    img3 = cv2.imread('iron_2.jpg')

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    kp3, des3 = sift.detectAndCompute(img3, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
    search_params = dict()   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    matches2 = flann.knnMatch(des1, des3, k=2)

    points = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            points.append(kp2[m.trainIdx].pt)
            
    avarage_point1 = get_avarage_point(points)
            
    points2 = []
    for m,n in matches2:
        if m.distance < 0.7*n.distance:
            points2.append(kp3[m.trainIdx].pt)
            
    avarage_point2 = get_avarage_point(points2)
    print(avarage_point1)
    print(avarage_point2)
    
    distance_between_cameras = 200
    focus_distance = 55
    distance_to_object = (distance_between_cameras*focus_distance)/(avarage_point1[0]-avarage_point2[0])
    distance_to_object2= (distance_between_cameras * 1280) / (math.atan(3)*(avarage_point1[0]-avarage_point2[0]))
    print("Расстояние до объекта 2 = " + str(distance_to_object2))

    print("Расстояние до объекта = " + str(distance_to_object))
            

            
    cv2.circle(img2, (int(avarage_point1[0]), int(avarage_point1[1])), 6, (244, 255, 0))
    
    cv2.circle(img3, (int(avarage_point2[0]), int(avarage_point2[1])), 6, (244, 255, 0))
    cv2.imshow("", img2)
    cv2.imshow("2", img3)
    cv2.waitKey(0)        


def get_avarage_point(points):
    sum_x = 0
    sum_y = 0
    for x,y in points:
        sum_x = sum_x + x
        sum_y = sum_y + y
    center_x = sum_x/len(points)
    center_y = sum_y/len(points)
    return (center_x, center_y)

if __name__=='__main__':
    main()

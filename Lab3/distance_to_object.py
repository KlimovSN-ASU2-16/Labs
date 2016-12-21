import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def main():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    img1 = cv2.imread('box.png')         
    img2 = cv2.imread('2м_2.JPG')

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()                                                                                                                                                                                                        

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(check = 50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)


    matches = flann.knnMatch(des1,des2,k=2)
    
    points = []
    good=[]
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            points.append(kp2[m.trainIdx].pt)
            
    if len(good)>10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts,  cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        
        h, w, d = img1.shape
        pts = np.float32([ [0,0], [0, h-1], [w-1, h-1], [w-1, 0] ]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
    
        width_box_pixels = dst[3][0][0]-dst[0][0][0]
        width_picture = len(img2[0])
        angle_camera = 60
        
        print(dst)
        print("width_box" + str(width_box_pixels))
        angle_box = width_box_pixels * angle_camera / width_picture
        
        print("angle_box " + str(angle_box))
        center_x = width_picture/2
        center_y = len(img2)/2
        
        print("center x " + str(center_x))

        dist1 =float(abs(dst[3][0][0]-center_x))
        dist2 = float(abs(dst[0][0][0]-center_x))
        print("dists " + str(dist1) + " " + str(dist2))
        distance_to_box_about_center = 0
        if dist1 > dist2:
            distance_to_box_about_center = dist2
        else:
            distance_to_box_about_center = dist1
        
        print("distance_to_box_about_center" + str(distance_to_box_about_center))
        
        
        angle_not_box = distance_to_box_about_center*angle_camera/width_picture
        print("angle_not_box" + str(angle_not_box))
        
        shared_angle = angle_box+angle_not_box
        print("shared_angle " + str(shared_angle))
        
        width_box_mm = 225
        distance_to_box_about_center_real= distance_to_box_about_center*width_box_mm/width_box_pixels
        print("distance_to_box_about_center_real " + str(distance_to_box_about_center_real))
        
        
        width = width_box_mm+ distance_to_box_about_center_real
        
        print("widht " + str(width))
        
        shared_angle_radians = shared_angle*math.pi/180
        distance_to_object=width/math.tan(shared_angle_radians)
        print(round(distance_to_object/10, 2))

        img2 = cv2.polylines(img2, [np.int32(dst)],  True, 255, 3,  cv2.LINE_AA)
    
        cv2.imshow(" ", img2)
            
            
            
            
            
            
            
            
            
    
    cv2.waitKey(0)        


if __name__=='__main__':
    main()

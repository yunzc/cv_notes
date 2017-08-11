# Yun Chang CV notes part 1
# Edge detections and Hough transform 

import cv2 
import numpy as np 

"""
Edge detection is done by calculating the gradient then thresholding the calculated gradient 
and keeping the places where the gradient is above a certain value. In other words, edges are 
where there are significant change in pixel value (talking gray-scale) where the significance 
is determined by coder. The often used Canny filter Gaussian filters the picture before gradient
calculation -> but by linear properties (associative) it turns out to be just filtering with the 
derivative of the gaussian. Canny: 
1. filter with derivative of gaussian 
2. threshold to find significant gradient area 
3. thinning (find the maximum in a local area) 
4. linking (if there are weak pixels linking strong pixels: edge)

most common kernel used to find gradient (discrete): Sobel
"""

# img = cv2.imread('stata.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # arg Canny(image, low_thresh, high_thresh, sobel_kernel_size)
# edges = cv2.Canny(gray,100,200,5)

# cv2.imshow('edges', edges)

# cv2.waitKey(0)

"""
Hough transform is based on the concept of voting: easiest way to think about it is such: let's say 
the normal space is x y space, now we can characterize each line by d the closest distance from line 
to origin, and theta the angle between d and the x axis. x*cos(theta) + y*sin(theta) = d; now we define 
a Hough space of d and theta. A line in xy space is now a point in Hough space, and a point in Hough 
space is now sinusoids. So we take each points in the edge image, "plot" them in Hough space, and find 
the intersections to the sinusoids (the maximums) to find the points that define lines in xy space. 
"""

# # implementation with cv2.HoughLines 
# lines = cv2.HoughLines(edges,1,np.pi/180,175)
# # arg: (edge_img, d accuracy, theta accuracy, threshold)
# img1 = img
# for i in range(len(lines)):
# 	for rho,theta in lines[i]:
# 	    a = np.cos(theta)
# 	    b = np.sin(theta)
# 	    x0 = a*rho
# 	    y0 = b*rho
# 	    x1 = int(x0 + 1000*(-b))
# 	    y1 = int(y0 + 1000*(a))
# 	    x2 = int(x0 - 1000*(-b))
# 	    y2 = int(y0 - 1000*(a))
# 	    cv2.line(img1,(x1,y1),(x2,y2),(255,0,0),5)
# 	    print(x1,y1,x2,y2)
# cv2.imshow('HoughLines', img1)
# cv2.waitKey(0)

# to implement this from scratch, create array and discretize line to points. Add to an index of array everytime
# a line "hits" the point --> then find those above threshold 
# probabilistic Hough Line transform: HoughLinesP

"""
Similar idea for a circle: for each point vote for the points along the circle around the point with a certain 
radius (but this time have to iterate over different radii): polling, find max 
"""

# implementation with cv2.HoughCircles 

img2 = cv2.imread('pool.jpg')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray2 = cv2.medianBlur(gray2, 5)
circles = cv2.HoughCircles(gray2,cv2.HOUGH_GRADIENT,1,20,
                            param1=200,param2=45,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img2,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',img2)
cv2.waitKey(0)

"""
Generalized Hough transform for arbitrary shape 
First, generate the Hough table. Go around boundary of shape, pick a reference point, and 
store the displacement indexed by the angle. Sometimes for one angle there might be multiple displacements, 
just store all in table. The on subject image, go backwards and vote. (get lines usually) 
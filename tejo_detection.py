import cv2
import numpy as np

img = cv2.imread('imagenes/4.jpg',0)


height, width = img.shape

scale = 4
height, width = (int(width / scale), int(height/ scale))
img = cv2.resize(img, (height, width))
img = cv2.GaussianBlur(img,(5, 5),5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.2, 200,
                            param1=40,param2=34,minRadius=50,maxRadius=90)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

img = cv2.imread('./Images/3j4c.png',0) 

# image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
# img = cv2.adaptiveThreshold(img,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

cv2.imshow("OpenCV Image Reading", img)
cv2.waitKey(0)
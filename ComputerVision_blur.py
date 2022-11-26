import cv2
import numpy as np


original_image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

kernel = np.ones((5,5))/25

blur_image = cv2.filter2D(original_image,-1, kernel)    # -> ddepth = -1, i.e. depth of the blur image is same as depth of original image

# gaussian blur is used to reduce noise !!!

cv2.imshow('Original Image', original_image)
cv2.imshow('Blurred Image', blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(kernel)
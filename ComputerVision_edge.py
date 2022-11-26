import cv2
import numpy as np


original_image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)

# we have to transform the image into grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# create a Laplacian kernel
kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])



result_image = cv2.filter2D(gray_image, -1, kernel)
# also we can use inbuilt laplacian
#result_image = cv2.Laplacian(gray_image, -1)

cv2.imshow('Original Image', gray_image)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(kernel)
import cv2
import numpy as np


original_image = cv2.imread('unsharp_bird.jpg', cv2.IMREAD_COLOR)



# create a sharpen kernel
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

# face recognition and we have the blurry CCTV video then we can apply this kernel in order to
# increase the precision of the underlying model

result_image = cv2.filter2D(original_image, -1, kernel)
# also we can use inbuilt laplacian
#result_image = cv2.Laplacian(gray_image, -1)

cv2.imshow('Original Image', original_image)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

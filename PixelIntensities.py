import cv2
import numpy as np
# handling grayscale images

# image = cv2.imread('camus.jpg', cv2.IMREAD_GRAYSCALE)


# # values close to 0 - darker
# # values close to 255 - brighter
# print(image)
# print(image.shape)

# cv2.imshow('Computer_Vision', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# handling coloured images

image = cv2.imread('bird.jpg', cv2.IMREAD_COLOR)


# values close to 0 - darker
# values close to 255 - brighter

# we store the R and G and B on 8 bits
print(np.amax(image))
print(image.shape)

cv2.imshow('Computer_Vision', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

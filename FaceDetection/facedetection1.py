import cv2
import matplotlib.pyplot as plt 


# PARAMETERS FOR FACE DETECTION 
# 
#  cascadeClassifier.detectMultiScale(image, faceDetections, scaleFactor, minNeighbors, flags, minSize, maxSize)
#
# 1.) scaleFactor: Since some faces may be closer to the camera, they would appear bigger
# 		 than other faces in the background -> the scale factor compensates for this
#			
#			Specifying how much the image size is reduced at each image scale
# 			
# 			The model has a fixed size defined during training: in the haarcascade_frontalface_alt.xml file !!!
# 			By rescaling the input image, you can resize a larger face to a smaller one,
# 				 making it detectable by the algorithm
# 
# 		Value: 1.1 - 1.4
# 			Small -> algorithm will be slow since it is more thorough
# 			High -> faster detection with the risk of missing some faces altogether
# 
#  2.) minNeighbors: specifying how many neighbors each candidate rectangle should have to retain it
#  			Value interval: ~ 3-6
#  				Higher values -> less detections but with higher quality !!!
#  
#  3.) flags: kind of a heuristic
#  		Reject some image regions that contain too few or too much edges
#  			 and thus can not contain the searched object
#  
#  4.) minSize: objects smaller than that are ignored !!!
#  			We can specify what is the smallest object we want to recognize 
#  					[30x30] is the standard
#  
#  5.) maxSize: objects larger than that are ignored !!!

# OpenCV has a lot of pre-trained classifiers for face detection, eye detection etc
# several positive and negative samples to train the model (Viola - jones algorithm)

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

image = cv2.imread('girl_image.jpg')



# transforming hte orginal image into gray scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

for (x,y, width, height) in detected_faces:
    cv2.rectangle(image, (x,y), (x+width, y+height), (0,0,255), 1)

# OpenCV uses BGR and matplotlib uses RGB
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

import cv2 

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# we will use real-time video (camera) - 0 means open the default camera
video_capture = cv2.VideoCapture(0)

# setting up the width and height of the video window
video_capture.set(3, 640)  # 3 -> width
video_capture.set(4, 480)  # 4 -> width

while True:

    # return the next video frame(the img is the important)
    ret, img = video_capture.read()

    # transforming hte orginal image into gray scale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x,y, width, height) in detected_faces:
        cv2.rectangle(img, (x,y), (x+width, y+height), (0,0,255), 10)

    # title of the video window
    cv2.imshow('Real time face detection', img)

    # we wait for a key to be pressed - press 'ESC' to quit
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

# destroy and release the camera etc
video_capture.release()
cv2.destroyAllwindows()



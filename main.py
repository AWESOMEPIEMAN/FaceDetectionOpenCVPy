import cv2
# cascade
face_cascade = cv2.CascadeClassifier('face_detector.xml')
eye_detector = cv2.CascadeClassifier('eye_detector.xml')
smile_detector = cv2.CascadeClassifier('smile_detector.xml')
# to open webcam
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_detector.detectMultiScale(gray,1.2,4)
    smile = smile_detector.detectMultiScale(gray,1.1,4)
    # region based box = face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    for (x,y,w,h) in smile:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Display
    cv2.imshow('Face', img)
    # escape key stops the progam
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release videocapture
cap.release()

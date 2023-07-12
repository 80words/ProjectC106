import cv2


# Create our body classifier
face_cascade=cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')


# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    people= face_cascade.detectMultiScale(gray, 1.2, 3)
    #Convert Each Frame into Grayscale
    
    for (x,y,w,h) in people:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (102,109,209), 2)
    # Pass frame to our body classifier
    
    cv2.imshow("cam", frame)
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()

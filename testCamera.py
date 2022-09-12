import cv2
from utils.dataloaders import LoadImages
live = LoadImages(path="./datasets/Examples/Sequences/2022-07-02-213312",img_size=448,stride=32,auto=False,reverse=False)
b = live[10:10000]
print(len(b))

#video capture object
cap=cv2.VideoCapture(0) #iphone camera

# capture the frames..
while True:
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)  # Display the resulting frame
    key = cv2.waitKey(1)
    if key == 27:  # click esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
# # Open the camera
# adress = "rtps://ezhomecam:LipM6Gqm4d@192.168.68.60/1"
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#     cv2.imshow('Input', frame)

#     c = cv2.waitKey(1)
#     if c == 27:
#         break
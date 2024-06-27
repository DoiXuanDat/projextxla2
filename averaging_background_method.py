import cv2
import numpy as np

cap = cv2.VideoCapture('IMG_2507.MOV')

# read the frames from the camera
ret, img = cap.read()

averageValue = np.float32(img)

# loop runs if capturing has been initialized.
while True:
    # reads frames from a camera
    ret, img = cap.read()

    # using the cv2.accumulateWeighted() function
    # that updates the running average
    cv2.accumulateWeighted(img, averageValue, 0.02)

    # converting the matrix elements to absolute values
    # and converting the result to 8-bit.
    resultingFrames = cv2.convertScaleAbs(averageValue)

    # Show two output windows
    # the input / original frames window
    cv2.imshow('Input', img)

    # the window showing output of alpha value 0.02
    cv2.imshow('average', resultingFrames)

    # Wait for Esc key to stop the program
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()